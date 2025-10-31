# Limitations of Pytorch on Cerebras - Cerebras AI

Developing for the Cerebras Wafer-Scale Cluster requires a different approach to PyTorch programming. The key is to think in terms of static, tensor-based computational graphs rather than dynamic, Python-driven logic.

## Floating Point Precision

The Cerebras system supports only mixed-precision training:

- Weights are stored as float32
- Computations use a combination of float32 and one of:
  - cbfloat16
  - bfloat16
  - float16

Precision casts are automatically inserted, so developers don’t need to manage them manually. See numeric precision for more information on switching between precision optimization levels.

## Ahead-of-Time (AOT) Compilation

The Cerebras Wafer-Scale Cluster relies on Ahead-of-Time compilation, which means:

- The entire model’s execution graph must be traced and compiled before execution
- Models run asynchronously
- PyTorch receives updates through callback mechanisms called step closures

## Coding Constraints and Recommendations
### Prioritize Tensor Operations

Avoid Python flow control like if statements and loops, as they can lead to unexpected behavior due to incomplete graph capture. Instead, use tensor operations like torch.where for conditional logic. The torch.where function acts as a tensor-based conditional, equivalent to an if statement but represented within the graph. Avoid any flow control that could cause a different graph. Only tensor operations are traced during execution. So the following code snippet:

```python
def bad_function(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    global step  # regular python int. global is bad for other reasons, but ignore that
    step += 1
    if step < 10:
        return input1
    else:
        return input2
```

would behave as if it were:

```python
def bad_function(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor
   return input1
```

Tracing captures only the first step’s static graph, necessitating tensor-based control flow. Instead, all operations must be tensors:

```python
def better_function(input1: torch.Tensor, input2: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
    step += 1
    output = torch.where(step < 10, input1, input2)
    return output
```

### Prematurely Fetching Tensor Values

During tracing, tensor values are not actually computed. As such, eagerly retrieving the value of tensors is not allowed. You must design your code to avoid accessing tensor values during tracing. For debugging and accessing tensor values, refer to the Step closures documentation. Look at the following example:

```python
def bad_function(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    if input1.max() < 0:
        raise ValueError("input1 must have positive elements!")
    return input1 + input2
```

During tracing, the value of input1.max() is not known but is required to compute the Python conditional. Hence, this will lead to the following tracing error:

```shell
  Traceback (most recent call last):
    File "modelzoo/fc_mnist/pytorch/model.py", line 50, in forward
      pred_logits = bad_function(self.last_layer(x))
    File "modelzoo/fc_mnist/pytorch/model.py", line 11, in bad_function
      if input1.max() < 0:
  RuntimeError: The Cerebras backend does not yet support `aten::_local_scalar_dense(Tensor self) -> Scalar`
```

The other commonly used statements that could lead to evaluating tensors are printing, asserting the tensor value, and calling tensor.to("cpu"), and calling tensor.item().

### Avoid Decaying Traced Tensors to Scalars

Tensors generated during tracing have their values hidden until execution, as the Cerebras WSC compiles the model graph before running it. Certain torch operations expect scalar values as arguments, but passing traced tensors to them can lead to unexpected behavior. Always explicitly convert traced tensors to scalars before using them in these contexts. Try restructuring code to avoid traced tensors in scalar operations whenever possible, as explicit conversions can impact performance. Torch operations such as torch.add (alpha argument expects a scalar, and not a traced tensor) and torch.addcdiv (which expects a scalar) can offer performance benefits. They leverage fused-multiply-add or similar techniques for efficiency. However, remember that they expect true scalar values for certain arguments (e.g., value, alpha):

```python
output = input.addcdiv(tensor1, tensor2, 3.14)
```

Constants or tensors created outside tracing work as expected, as their values are readily available:

```python
pi = torch.tensor(3.14, dtype=torch.float16)
output = input.addcdiv(tensor1, tensor2, pi)
```

While scalars and external tensors work seamlessly, those generated within traced operations (like learning rate schedules) can cause trouble:

```python
lrs = torch.where(step > 1000, step.float() * 0.0001, 0.0002)
output = input.addcdiv(tensor1, tensor2, lrs)
```

The implicit Tensor > Scalar decay will trigger the SyncTensor called outside of MarkStep error and a random value will be used instead. In addition, if this was the only “use” of the lrs tensor, walking the dependency graph of all traced values will show no computation depending on the value of lrs and the entire set of tensor operations will not be lowered and compiled down.

### Pre-Initialize Stateful Tensors

The Cerebras WSC compiles the model’s execution graph based on operations observed during the first step, assuming subsequent steps follow the same pattern. Tensors carry state information across steps and are essential for optimizers and other stateful components. The Cerebras WSC tracks the identities of stateful tensors to maintain their values between steps. Initializing stateful tensors within conditional blocks or based on runtime conditions during the first step can lead to issues. Therefore, ensure all stateful tensors are explicitly initialized before tracing begins, even if their initial values might be overwritten later. If a stateful tensor isn’t initialized before tracing begins, it won’t have a registered identity within the system. As a result, the Cerebras WSC won’t recognize it as a persistent tensor that needs to be loop-carried. In each subsequent step, the tensor will be treated as a new tensor and initialized to zero, losing any previously updated values. Instead of being loop-carried, the updated tensor values get misinterpreted as model outputs like predictions or losses. Because the Cerebras WSE compiles the model’s execution graph based on the operations observed during the first step, assuming all subsequent steps will follow the same pattern, any stateful tensors (those that maintain state across steps) must be initialized explicitly before tracing begins. Initializing them conditionally or during runtime within the first step can lead to tracking issues and unexpected behavior and should be avoided:

```python
class CustomOptimizer(torch.optim.Optimizer):
    ...
    def step(self):
        ...
        for param_group in self.param_groups:
            for p in param_group["params"]:
                if len(self.state[p]) == 0:
                    state[p]["momentum"] = torch.zeros_like(p.grad)
                state[p]["momentum"].add_(p.grad * 0.001)
        ...
```

In addition to tracing operations, we also track the “identity” of certain tensors in order to loop-carry state between successive steps of the computation - the updated valued for state[param]["momentum"] on step 1 should be fed as the initial value for state[param]["momentum"] on step 2 etc. These values are “weights” and kept resident in system memory of some kind (on wafer in pipeline mode, or in the MemoryX/weight hosts in weight streaming). In this case (even disregarding the python flow control), the state[param]["momemtum"] has no identity before this first step. Executing the operations recorded by this trace would re-initialize momentum to zero on every single step, with the updated tensor result having no place to be stored (it would be treated as another model output like loss or predictions).

### Optimizer Considerations

Following the above guidelines and leveraging the Cerebras optimizer wrapper capabilities, you can seamlessly integrate your PyTorch optimizers with the Cerebras Wafer Scale Cluster. Cerebras optimizer wrapper’s preinitialize() function ensures that all stateful tensors within the optimizer, typically stored in its state_dict, are explicitly initialized before the tracing process begins. This is crucial for Cerebras WSC compatibility. The Cerebras framework automatically calls preinitialize() at the appropriate time, simplifying compliance. The function can also be called within the optimizer’s _init_ method for traditional GPU-based training, providing a flexible implementation. The Cerebras optimizer wrapper injects a traced tensor for param_group["lr"] to dynamically implement learning rate schedules based on a traced global_step tensor. Passing these traced tensors to operations that expect scalar values can lead to unexpected behavior or errors due to implicit decay attempts. Refactor static graphs by exploring alternative tensor-based implementations or restructuring to avoid non-static graph constructs.

## Learning Rate Scheduler

Currently, we do not support the typical PyTorch learning rate scheduler paradigm. A typical PyTorch learning scheduler would compute a learning rate scalar and set the values of the learning rates in the optimizer parameter groups. However, we cannot support this behavior due to the system’s current limitations requiring static graphs. We must specify the entire learning rate schedule as a function of the global step. This means that the learning rate becomes less of a scalar value and more of a tensor that depends on the value of the global step. See Learning rate scheduling for more details. This also means that any optimizers being used need to be written in a way such that the learning rate is not treated as a scalar value but rather as a tensor. See cerebras.pytorch.optim.AdamBase for an example of this.

## Multiple Execution Limitations

The cluster configuration is locked in on the first compile job encountered. This means the first compile should be the most resource-intensive job. That is to say, if both training and eval are being performed, training must come before eval. If eval is performed before training, the train compile might fail if the first eval compile didn’t allocate enough resources. This also means CSConfig configurations, such as num_csx and num_workers_per_csx are global across all executions.

### Modifying Stateful Tensors Between Executions

Stateful tensors may be modified between executions if a checkpoint was taken on the last step of the previous execution. For example, between a training and evaluation run, changing model parameters or optimizer states for instance is only supported if a checkpoint was taken in the last training step. What this means is that checkpoint steps must be greater than 0 in the data executor:

```python
train_executor = cstorch.utils.data.DataExecutor(
    num_steps=1000, checkpoint_steps=100
)
for inputs, targets in train_executor:
    ...

model.fc1.weight.data.mul_(0.9)  # This is allowed

train_executor = cstorch.utils.data.DataExecutor(
    num_steps=1000, checkpoint_steps=0
)
for inputs, targets in train_executor:
    ...

model.fc1.weight.data.mul_(0.9)  # This is *not* allowed

validation_executor = cstorch.utils.data.DataExecutor(...)
for step, (inputs, targets) in enumerate(validation_executor):
    ...
```
