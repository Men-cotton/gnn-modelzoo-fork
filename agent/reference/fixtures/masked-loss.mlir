// Equivalent canonicalized loss/backward graph, authored for local lowering.
// %mask is the final supervision mask (target_mask AND labels != -100).
// This is not a full Python-to-CSX export.
module {
 func @graph(%logits: !torch.vtensor<[4,2],f32>, %labels: !torch.vtensor<[4],si64>, %mask: !torch.vtensor<[4],i1>, %grad: !torch.vtensor<[],f32>) -> (!torch.vtensor<[],f32>, !torch.vtensor<[4,2],f32>) {
   %none = torch.constant.none
   %zero = torch.constant.int 0
   %one = torch.constant.int 1
   %six = torch.constant.int 6
   %ignore = torch.constant.int -100
   %false = torch.constant.bool false
   %floatone = torch.vtensor.literal(dense<1.0> : tensor<f32>) : !torch.vtensor<[],f32>
   %zero_labels = torch.vtensor.literal(dense<0> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
   %safe_labels = torch.aten.where.self %mask, %labels, %zero_labels : !torch.vtensor<[4],i1>, !torch.vtensor<[4],si64>, !torch.vtensor<[4],si64> -> !torch.vtensor<[4],si64>
   %weights = torch.aten._to_copy %mask, %six, %zero, %none, %none, %false, %none : !torch.vtensor<[4],i1>, !torch.int, !torch.int, !torch.none, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[4],f32>
   %count = torch.aten.sum %weights, %none : !torch.vtensor<[4],f32>, !torch.none -> !torch.vtensor<[],f32>
   %denom = torch.aten.clamp_min.Tensor %count, %floatone : !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
   %log_probs = torch.aten._log_softmax %logits, %one, %false : !torch.vtensor<[4,2],f32>, !torch.int, !torch.bool -> !torch.vtensor<[4,2],f32>
   %losses, %total_weight = torch.aten.nll_loss_forward %log_probs, %safe_labels, %none, %zero, %ignore : !torch.vtensor<[4,2],f32>, !torch.vtensor<[4],si64>, !torch.none, !torch.int, !torch.int -> !torch.vtensor<[4],f32>, !torch.vtensor<[],f32>
   %masked = torch.aten.mul.Tensor %losses, %weights : !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
   %sum = torch.aten.sum %masked, %none : !torch.vtensor<[4],f32>, !torch.none -> !torch.vtensor<[],f32>
   %loss = torch.aten.div.Tensor %sum, %denom : !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
   %grad_scale = torch.aten.div.Tensor %grad, %denom : !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
   %grad_targets = torch.aten.mul.Tensor %weights, %grad_scale : !torch.vtensor<[4],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[4],f32>
   %nll_grad = torch.aten.nll_loss_backward %grad_targets, %log_probs, %safe_labels, %none, %zero, %ignore, %total_weight : !torch.vtensor<[4],f32>, !torch.vtensor<[4,2],f32>, !torch.vtensor<[4],si64>, !torch.none, !torch.int, !torch.int, !torch.vtensor<[],f32> -> !torch.vtensor<[4,2],f32>
   %logit_grad = torch.aten._log_softmax_backward_data %nll_grad, %log_probs, %one, %six : !torch.vtensor<[4,2],f32>, !torch.vtensor<[4,2],f32>, !torch.int, !torch.int -> !torch.vtensor<[4,2],f32>
   return %loss, %logit_grad : !torch.vtensor<[],f32>, !torch.vtensor<[4,2],f32>
 }
}
