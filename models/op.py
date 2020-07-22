# pylint:disable-all
import warnings
import weakref

import numpy as np
import scipy.optimize as sop
from scipy.special import logsumexp, softmax
import torch
from torch import nn
import torch.nn.functional as F

# ---- S-FUNC: sigmoid_log ----
def sigmoid_log(w, beta_1, beta_2):
    # Should use (filters_l1 / self.beta_1) ** self.beta_2
    # maybe this way, the parametrization would be relatively more independent,
    # and beta_2 could reflect the variance better.
    # More specifically, when beta_2 is adjusted,
    # beta_1 do not need large adjustment relatively to make clipped value smaller.
    # This might benefit the faster adjustment/correction of beta_1?
    # implicit relation is Normalization(f(max_i |W_i|, \beta_1, \beta_2)) = 1/C'.
    # how to organize to make \frac{\partial \beta_1}{\partial \beta_2}

    # Considering the limit case.
    # As \beta_2 -> \inf, the soft thresholding become closer to hard thresholding.
    # And beta_1 is the hard threshold for |W|:
    # * |W| < beta_1 -> score = 0
    # * |W| > beta_1 -> score = \inf
    # No matter how \beta_2 is adjusted in this regime to reduce the variance,
    # beta_1 do not need to change.
    # score = (w / beta_1) ** beta_2
    # return score / (score + 1)

    # However, the beta_1 actually have implicit gradient regards to prune ratio
    # and it might matters more than we think. Could we really decoule alpha and beta_1/beta_2?
    # Is current instruction from the loss to alpha already sufficient.

    # might be more numerically stable
    inv_score = (w / beta_1) ** (-beta_2)
    return 1 / (1 + inv_score)

def sigmoid_log_lbeta1(w, log_beta1, beta2):
    EPS = 1e-8
    tmp = log_beta1 - torch.log(w + EPS)
    tmp = torch.clamp(tmp * beta2, min=-20., max=20.)
    return torch.clamp(1. / (1 + torch.exp(tmp)) + 0.01, min=0., max=1.)

class BernoulliSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probs):
        dist = torch.distributions.Bernoulli(probs)
        return dist.sample()

    @staticmethod
    def backward(ctx, grads):
        return grads

class SigmoidLogMergeFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base_imp, alpha, beta_2, ori_beta1=None):
        base_imp_detach = base_imp.detach().cpu()
        alpha_detach = alpha.detach().cpu()
        beta2_detach = beta_2.detach().cpu()
        base_imp_np = base_imp_detach.numpy()
        alpha_np = alpha_detach.numpy()
        beta2_np = beta2_detach.numpy()
        if ori_beta1 is None:
            init_guess = (1 - alpha_np) * base_imp_np.max()
        else:
            init_guess = ori_beta1.item()
        C = base_imp.shape[0]
        def func(beta1):
            EPS = 1e-8
            beta1 = np.maximum(beta1, EPS)
            tmp = np.log(beta1 + EPS) - np.log(base_imp_np + EPS)
            tmp = np.minimum(tmp * beta2_np, 10.)
            return (1. / (1 + np.exp(tmp))).mean() - alpha_np + 0.01
        def func_order1(beta1):
            EPS = 1e-8
            beta1 = np.maximum(beta1, EPS)
            tmp = beta2_np * (np.log(beta1 + EPS) - np.log(base_imp_np + EPS))
            return - np.exp(
                logsumexp(np.log(beta2_np + EPS) - np.log(beta1 + EPS) \
                          + tmp - 2 * (np.log(1 + np.exp(-np.abs(tmp))) + np.maximum(tmp, 0)) - np.log(C)))
        try:
            beta1_res = sop.root_scalar(
                func, x0=init_guess, bracket=[1e-10, base_imp_np.max() * 10], fprime=func_order1, rtol=0.02)
            beta1 = np.maximum(beta1_res.root, 1e-10)
        except RuntimeError as e:
            print(e)
        beta1 = alpha.new([beta1])
        ctx.save_for_backward(base_imp_detach, alpha_detach, beta2_detach, beta1)
        return sigmoid_log(base_imp, beta1, beta_2), beta1.detach()

    @staticmethod
    def backward(ctx, grad_output, _):
        base_imp_detach, _, beta2_detach, beta1 = ctx.saved_tensors
        base_imp_np = base_imp_detach.numpy()
        beta1_np = beta1.cpu().detach().numpy()
        beta2_np = beta2_detach.numpy()
        C = base_imp_detach.shape[0]
        EPS = 1.e-8
        tmp = beta2_np * (np.log(beta1_np + EPS) - np.log(base_imp_np + EPS))
        log_f_prime = tmp - 2 * (np.log(1+np.exp(-np.abs(tmp))) + np.maximum(tmp, 0))
        normalized_f_prime = grad_output.new(softmax(log_f_prime))
        return None, (grad_output * normalized_f_prime).squeeze().sum(dim=0, keepdim=True) * C, None, None

class SigmoidLogMergeFunc_subst(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base_imp, alpha, beta_2, ori_beta1=None):
        base_imp_detach = base_imp.detach().cpu()
        alpha_detach = alpha.detach().cpu()
        beta2_detach = beta_2.detach().cpu()
        base_imp_np = base_imp_detach.numpy()
        alpha_np = alpha_detach.numpy()
        beta2_np = beta2_detach.numpy()
        if ori_beta1 is None:
            init_guess = (1 - (alpha_np - 0.01)) * base_imp_np.max()
        else:
            init_guess = ori_beta1.item()
        EPS = 1e-8
        init_guess = np.log(init_guess + EPS)
        C = base_imp.shape[0]

        def func(log_beta1):
            EPS = 1e-8
            tmp = log_beta1 - np.log(base_imp_np + EPS)
            tmp = np.clip(tmp * beta2_np, a_min=-20., a_max=20.)
            return np.minimum(1. / (1 + np.exp(tmp)) + 0.01, 1.).mean() - alpha_np
        def func_order1(log_beta1):
            EPS = 1e-8
            tmp = beta2_np * (log_beta1 - np.log(base_imp_np + EPS))
            return - np.exp(
                logsumexp(np.log(beta2_np + EPS) - log_beta1 \
                          + tmp - 2 * (np.log(1 + np.exp(-np.minimum(np.abs(tmp), 50.))) \
                                       + np.maximum(tmp, 0)) - np.log(C)) + log_beta1)
        try:
            log_beta1_res = sop.root_scalar(
                func, x0=init_guess, bracket=[-500., 500.], xtol=0.02)
            log_beta1 = log_beta1_res.root
            if not log_beta1_res.converged:
                raise Exception("not log_beta1_res converged!")
        except RuntimeError as e:
            print(e)
        log_beta1 = alpha.new([log_beta1])
        beta1 = log_beta1.exp()
        ctx.save_for_backward(base_imp_detach, alpha_detach, beta2_detach, beta1, log_beta1)
        return sigmoid_log_lbeta1(base_imp, log_beta1, beta_2), beta1.detach()

    @staticmethod
    def backward(ctx, grad_output, _):
        np.seterr(all="raise")
        try:
            base_imp_detach, _, beta2_detach, beta1, log_beta1 = ctx.saved_tensors
            base_imp_np = base_imp_detach.numpy()
            beta2_np = beta2_detach.numpy()
            log_beta1_np = log_beta1.detach().cpu().numpy()
            C = base_imp_detach.shape[0]
            EPS = 1.e-8
            tmp = beta2_np * (log_beta1_np - np.log(base_imp_np + EPS))
            log_f_prime = tmp - 2 * (np.log(1+np.exp(-np.minimum(np.abs(tmp), 50.))) \
                                     + np.maximum(tmp, 0))
            log_f_prime = np.maximum(log_f_prime - log_f_prime.max(), -50.)
            normalized_f_prime = grad_output.new(softmax(log_f_prime))
        except:
            raise
        return None, (grad_output * normalized_f_prime).squeeze().sum(dim=0, keepdim=True) * C, None, None

class SigmoidLogBeta1Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base_imp, alpha, beta2, ori_beta1=None):
        base_imp_detach = base_imp.detach().cpu()
        alpha_detach = alpha.detach().cpu()
        beta2_detach = beta2.detach().cpu()
        base_imp_np = base_imp_detach.numpy()
        alpha_np = alpha_detach.numpy()
        beta2_np = beta2_detach.numpy()
        if ori_beta1 is None:
            init_guess = (1 - alpha_np) * base_imp_np.max()
        else:
            init_guess = ori_beta1.item()
        C = base_imp.shape[0]
        def func(beta1):
            return (1. / (1 + (beta1 / base_imp_np) ** beta2_np)).mean() - alpha_np
        def func_order1(beta1):
            beta1 = np.maximum(beta1, EPS)
            tmp = (beta1 / base_imp_np) ** beta2_np
            return - (beta2_np / beta1 * tmp / (1 + tmp) ** 2).mean()
        try:
            beta1_res = sop.root_scalar(func, x0=init_guess, fprime=func_order1, xtol=0.02, method="newton")
            beta1 = np.maximum(beta1_res.root, 1e-10)
        except RuntimeError as e:
            print(e)
        beta1 = alpha.new([beta1])
        ctx.save_for_backward(base_imp_detach, alpha_detach, beta2_detach, beta1)
        return beta1

    @staticmethod
    def backward(ctx, grad_output):
        base_imp_detach, alpha_detach, beta2_detach, beta1 = ctx.saved_tensors
        base_imp_np = base_imp_detach.numpy()
        C = base_imp_np.shape[0]
        alpha_np = alpha_detach.numpy()
        beta2_np = beta2_detach.numpy()
        beta1_np = beta1.detach().cpu().numpy()
        EPS = 1e-8
        # tmp = np.log(beta1_np / (base_imp_np + EPS) + EPS)
        # beta2 big enough
        # partial_beta1_alpha = C * np.exp(-logsumexp(np.log(beta2_np / beta1_np + EPS) - beta2_np * tmp)
        tmp = (beta1_np / base_imp_np) ** beta2_np
        partial_beta1_alpha = - 1. / (beta2_np / beta1_np * tmp / (1 + tmp) ** 2).mean()
        return None, grad_output * grad_output.new([partial_beta1_alpha]), None, None

class SigmoidLogBeta1Module(nn.Module):
    def forward(self, *args):
        return SigmoidLogBeta1Func.apply(*args)

sigmoid_log.merge_func = SigmoidLogMergeFunc
sigmoid_log.merge_func_subst = SigmoidLogMergeFunc_subst
sigmoid_log.beta1_module = SigmoidLogBeta1Module
# ---- End S-FUNC: sigmoid_log

class MaskedConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, groups=1, affine=True,
                 base_importance_strategy="conv", has_relu=True, is_primal=True,
                 init_beta_1=1.0, init_beta_2=1.0, s_func_name="sigmoid_log",
                 straight_through_grad=False, detach_base_importance=True,
                 plan=1):
        super(MaskedConvBNReLU, self).__init__()
        assert base_importance_strategy in {"conv", "conv_bn", "bn"}
        assert plan in {1, 2}
        self.plan = plan
        self.base_importance_strategy = base_importance_strategy
        self.straight_through_grad = straight_through_grad
        self.detach_base_importance = detach_base_importance
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c_out, affine=affine)
        self.has_relu = has_relu
        self.is_primal = is_primal
        self.ref_model = None
        if has_relu:
            self.relu = nn.ReLU(inplace=False)
        # ---- for mask ---
        mask = torch.ones(c_out)
        self.register_buffer("mask", mask)
        self.invsigmoid_keep_ratio = nn.Parameter(torch.tensor([5.0]), requires_grad=True)
        # the original conv
        self.in_channels = c_in
        self.out_channels = c_out
        self.kernel_size = self.conv.kernel_size
        self.s_func_name = s_func_name
        self.s_func = globals()[self.s_func_name]
        assert init_beta_1 > 0
        assert init_beta_2 > 0
        # init
        invsp_beta_1 = np.log(np.exp(init_beta_1) - 1)
        invsp_beta_2 = np.log(np.exp(init_beta_2) - 1)
        self.invsp_beta_1 = nn.Parameter(torch.tensor([invsp_beta_1]))
        invsp_beta_2 = torch.tensor([invsp_beta_2])
        self.register_buffer("invsp_beta_2", invsp_beta_2)
        # temperature = torch.tensor([1.])
        temperature = torch.tensor([0.001])
        self.register_buffer("temperature", temperature)

        # ---- plan 1 ----
        self.s_func_beta1_module = self.s_func.beta1_module()

        # ---- for parsing ----
        self.o_size = None

    @property
    def keep_ratio(self):
        return torch.sigmoid(self.invsigmoid_keep_ratio)

    """
    Manage primal conv modules outside is better, and more controllable.
    This ref here cause problem when primal module is not the first one being forwarded,
    and in fact, get mask according to weights distribution or randomly (as in VI) is decoupled from the
    forward process, and should be done outside each forward. It should be done once for each components.

    def set_primal(self, primal=True):
        self.is_primal = primal

    def set_primal_ref(self, ref_mod):
        if ref_mod is self:
            self.set_primal()
            self.ref_model = None
        else:
            self.ref_model = weakref.ref(ref_mod)
    """

    def set_mask(self, mask):
        self.mask = mask

    def get_mask_and_prob(self):
        """
        Plan 2:
        *importance criterion*: L1 norm of each filter
        TODO: merge BN, handle skip connection. Maybe check criterion-related paper.

        s_i = f(w_i; \beta_1, \beta_2)
        \theta_i = \mbox{Normalize}(s_i, \{s_i\}_{i=1, \cdots, C})
        p_i = \mbox{Clip}(C' \theta_i, 1)
        m_i ~ \mbox{Bernoulli}(p_i)
        obviously, if p_i is clipped, the expectation of $\sum_{i=1}^C m_i$
        would deviate from the descired C' (smaller).
        this two layerwise global parameters \beta_1 \beta_2 are used to control
        the shape and location of the S-shaped func $f$ to minimize the variance and the clip.

        TODO: Actually, the CDF of an arbitrary uni-modal continous distribution is
        an S-shape function from the sample space to [0, 1]. Other choices can be tried.

        * The layerwise f(w; \beta_1, \beta_2) models the L1 norm -> relative importance.
        * Variance and smaller clip (expectation closer to the desired C') as regularization.
          To make the variance smaller, the S-shape relative-saliency function should be more steep.
          Note that \beta_1 and \beta_2 are treated as global hyper-parameters to be optimized at
          a slower pace, to adapt to current weight distribution and pruning ratio \alpha,
          according to the regularizations of the desired propoties of the "relative importance"
          over time.
          They would not participate as intermediate nodes in the backpropagate process of learning
          \alpha and weights. As we ``decouple'' the choices of \beta_1, \beta_2 from \alpha and
          current weight distribution using regularization-based tricks.

        ---- TODO ----
        A. Bayesian modeling of Plan1:
        regard p_i as independent r.v.s from Beta distribution.
        add an Bi-modal beta prior on p_i.
        Instead of controling the p_i directly by some function family.
        We could control p_i implicitly by using the prior to model our expectation:
        p_i \sim Beta(a, b) \in [0, 1]. satistify the B constraint naturally.
        Now we need to design a(\alpha) and b(\alpha) to reflect our hopes in this prior,
        we can omit the use of using |W| as the importance criterion, and rely on
        the cross-entropy loss itself to inference for the posteriors for different channels.
        1) bi-modal for lower variance:
           We need an informative prior actually.
           The prior should encounrage E[p_i(1-p_i)] to be small as the training goes on,
           intuitively, as a, b goes down, the Beta dist becomes bi-modal, and p_i becomes 0s and 1s
           thus the variance V is smaller.
           So we need an adjustable \beta_2, bigger -> smaller a, b, smaller variance;

           Formally, V = a/(a+b) - B(a+2, b)/B(a, b) = ab / (a+b) / (a+b+1).
           As a, b becomes much smaller than 1, V is getting smaller linearly.
        2) E[\sum m_i] = \sum E[p_i] = C'. As the prior of each channel is the same,
           a/(a+b) = \alpha.
        Based on the above analysis, a = \alpha * \beta_2, b = (1-\alpha) * beta_2
        KL-divergence is zero-forcing, so the posterior should match one modal and
        i think it would work. The problem is that the constraint of target prune ratio is
        further loosen.

        B. Bayesian modeling of Plan2:
        regard \{\theta_i\}_{i=1,\cdots,C} as r.v from Dirichlet distribution.
        How to restrict the clip probability: P(max_{i=1,\cdots,C}(\theta_i) > 1/C') -> 0
        And in the meantime, control the variance(). it's hard, as the dirichlet pdf is continous,
        if each P(\theta_i > 1/C') need to be close to 0, then P(\theta_i = 1/C') would be small.

        # ----
        Order by constraint exactness:
        * Plan 1: Prune criterion assumption: importance positive correlates with |W|'s magnitude.
              modeling absolute p_i using some s-func family of |W|, alpha and p_is are
              related using constraint A;
            *forward*: solve implicit equation and get corresponding beta_1(alpha, |W|),
              beta_2(alpha, |W|) to satisify the expect variance and constraint A
              (prune ratio expectation).
            *backward*: implicit differentiation to backpropgate into alpha, and maybe |W|.
          Or use a simplification: beta_2 is the controllable hyper-parameter
          (not the expect variance upperbound). Only fit beta_1 using prune ratio expectation
          constraint (constraint A).
        * Plan 2: Prune criterion assumption: importance positive correlates with |W|'s magnitude.
             modeling *relative* s_i using some s-func family of |W|, alpha and s_i's are
             related through regularization B (clip reduction), and (optionally) variance
             regularization (could also be simplified as a scheduled hyper-parameter)
           *forward*: beta_1 beta_2 are regarded as global "hyper"-parameters,
           the relation of these hyperparameters to prune ratio are decoupled.
           \frac{\partial L}{\partial \alpha} would not pass through s_i.
           *backward*: the alpha's gradients only comes from $p_i = \alpha \theta_i$, there is
           no further backpropgation into $\theta_i, s_i, \beta_1, \beta_2$.
        * Bayesian modeling of Plan 1: modeling absolute p_i from bayesian perspective,
          do not assume channel importance rule in the pruning process,
          channel importance is done by variational inference.
          \alpha and p_i's are related through the prior expectation,
          prior variance could also be controlled via prior parameters.
          PROBLEM: The prior expectation could be far away from the posterior expectation,
          although i think some bound can be derived.
          But without proper KL coefficient, the loose constraint through KL force might fail to
          constrain posterior.
          *backward*: Different from traditional bayesian methods, our prior has one degree of
             freedom to update that should be instructed by the loss.
             We hope this could find a better prune ratio trade-off across layers.

          Adding regularization on the posterior variational parameters seems to be not elegant.
          And other bayesian pruning studies could also incorporate this
          regularization. However, if the regluarization is on the posterior variational parameters,
          the prune ratio's update is not instructed by the loss.
        """
        if self.base_importance_strategy == "bn":
            b_c_imp = self.bn.weight.abs()
        elif self.base_importance_strategy == "conv_bn":
            mean = self.bn.running_mean
            var = self.bn.running_var
            conv_l1 = self.conv.weight.abs().sum(dim=[1, 2, 3])
            b_c_imp = (conv_l1 * bn_scale / torch.sqrt(var + bn_eps)).abs()
        elif self.base_importance_strategy == "conv":
            b_c_imp = self.conv.weight.abs().sum(dim=[1, 2, 3])
        if self.detach_base_importance:
            b_c_imp = b_c_imp.detach()

        self.beta_2 = torch.nn.functional.softplus(self.invsp_beta_2)
        beta_1_init = getattr(self, "beta_1", None)
        if beta_1_init is not None:
            beta_1_init = beta_1_init.detach()
        # self.beta_1 = self.s_func_beta1_module(b_c_imp, self.keep_ratio, self.beta_2, beta_1_init)
        # probs = abs_saliency = self.s_func(b_c_imp, self.beta_1, self.beta_2)
        # abs_saliency, self.beta_1 = self.s_func.merge_func.apply(b_c_imp, self.keep_ratio, self.beta_2, beta_1_init)
        abs_saliency, self.beta_1 = self.s_func.merge_func_subst.apply(b_c_imp, self.keep_ratio, self.beta_2, beta_1_init)
        self.probs = probs = abs_saliency
        expectation_diff = abs_saliency.mean() - self.keep_ratio.detach()

        # relaxed_bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(self.temperature, probs=probs)
        # masks = relaxed_bernoulli.rsample() # reparametrization gradients
        masks = BernoulliSample.apply(probs) # straight through gradients
        def _save_last_grad(grad):
            self.prob_grad = grad
        probs.register_hook(_save_last_grad)
        def _save_last_mask_grad(grad):
            self.mask_grad = grad
        masks.register_hook(_save_last_mask_grad)
        return masks, probs, expectation_diff

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        # mask behind convbn here
        if not self.straight_through_grad:
            out = out * self.mask.reshape([1, -1, 1, 1])
        else:
            # note this would straight through the gradients to the previous features and layers
            # if want to straight through to the weights of this layer, which is more reasonable,
            # (a local recoverable signal), should multiply the mask onto the conv weights
            out = out + out.detach() * self.mask.reshape([1, -1, 1, 1]) - out.detach()
            # out = StraightThroughMul.apply(out, self.mask.reshape([1, -1, 1, 1]))

        if self.has_relu:
            out = self.relu(out)
        if self.o_size is None:
            self.o_size = tuple(out.size())
        return out

class DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__() 
        assert stride == 2
        self.out_channels = nOut
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)   
  
    def forward(self, x):   
        x = self.avg(x)  
        if self.out_channels-x.size(1) > 0:
            return torch.cat((x, torch.zeros(x.size(0), self.out_channels-x.size(1), x.size(2), x.size(3), device=x.device)), 1)
        else:
            return x

class MaskedResBlock(nn.Module):
    expansion = 1

    def __init__(self, c_in, c_out, stride, drop_prob=None, kernel_size=3,
                 downsample="conv", affine=True, **kwargs):
        super(MaskedResBlock, self).__init__()
        assert downsample in {"conv", "avgpool", "conv2"}
        self.downsample = downsample
        self.drop_prob = drop_prob
        self.stride = stride
        padding = (kernel_size - 1) // 2
        self.op_1 = MaskedConvBNReLU(c_in, c_out, kernel_size, stride, padding, affine=affine, has_relu=True, **kwargs)
        self.op_2 = MaskedConvBNReLU(c_out, c_out, kernel_size, 1, padding, affine=affine, has_relu=False, **kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.is_skip_conv = not (stride == 1 and c_in == c_out)
        if self.is_skip_conv:
            if self.downsample == "conv":
                self.skip_op = MaskedConvBNReLU(c_in, c_out, 1, stride, 0,
                                                affine=affine, has_relu=False, **kwargs)
            elif self.downsample == "avgpool":
                self.skip_op = DownsampleA(c_in, c_out, stride)
            elif self.downsample == "conv2":
                self.skip_op = MaskedConvBNReLU(c_in, c_out, 2, stride, 0,
                                                affine=affine, has_relu=False)
        else:
            self.skip_op = nn.Identity()

    def forward(self, x):
        p = np.random.uniform()
        if self.drop_prob is not None and self.training and p < self.drop_prob.item():
            return x
        else:
            return self._forward(x)

    def _forward(self, x):
        out = self.op_1(x)
        out = self.op_2(out)
        skip_out = self.skip_op(x)
        return self.relu(out + skip_out)


class MaskedBottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, c_in,c_out, stride, drop_prob=None, kernel_size=3,
                 affine=True , **kwargs):
        super(MaskedBottleneckBlock, self).__init__()
        self.expansion = 4
        self.drop_prob = drop_prob
        self.stride = stride
        padding = (kernel_size - 1) // 2
        self.op_1 = MaskedConvBNReLU(c_in, c_out, 1, 1, 0, affine = affine, has_relu = True, **kwargs)
        self.op_2 = MaskedConvBNReLU(c_out, c_out, kernel_size, stride, padding, affine = affine, has_relu = True, **kwargs)
        self.op_3 = MaskedConvBNReLU(c_out, c_out*self.expansion, 1, 1, 0, affine = affine, has_relu = False, **kwargs)

        self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.is_skip_conv = not (stride == 1 and c_in == c_out*self.expansion)
        self.skip_op = MaskedConvBNReLU(c_in, c_out*self.expansion, 1, stride, 0,
                                        affine=affine, has_relu=False, **kwargs) \
                                        if self.is_skip_conv else nn.Identity()

    def forward(self, x):
        p = np.random.uniform()
        if self.drop_prob is not None and self.training and p < self.drop_prob.item():
            return x
        else:
            return self._forward(x)

    def _forward(self, x):
        out = self.op_1(x)
        out = self.op_2(out)
        out = self.op_3(out)
        skip_out = self.skip_op(x)
        return self.relu(out + skip_out)

