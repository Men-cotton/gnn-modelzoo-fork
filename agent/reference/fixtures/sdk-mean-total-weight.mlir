module {
 func @graph(%logits: !torch.vtensor<[4,2],f32>, %labels: !torch.vtensor<[4],si64>) -> (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) {
   %none = torch.constant.none
   %one = torch.constant.int 1
   %ignore = torch.constant.int -100
   %false = torch.constant.bool false
   %log_probs = torch.aten._log_softmax %logits, %one, %false : !torch.vtensor<[4,2],f32>, !torch.int, !torch.bool -> !torch.vtensor<[4,2],f32>
   %output, %total_weight = torch.aten.nll_loss_forward %log_probs, %labels, %none, %one, %ignore : !torch.vtensor<[4,2],f32>, !torch.vtensor<[4],si64>, !torch.none, !torch.int, !torch.int -> !torch.vtensor<[],f32>, !torch.vtensor<[],f32>
   return %output, %total_weight : !torch.vtensor<[],f32>, !torch.vtensor<[],f32>
 }
}
