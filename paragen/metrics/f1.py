from paragen.metrics import PairwiseMetric, register_metric


@register_metric
class F1(PairwiseMetric):
    """
    F1 evaluates F1 of produced hypotheses labels by comparing with references.
    """

    def __init__(self, target_label, is_labeling=False):
        super().__init__()
        self._target_label = target_label
        self._is_labeling = is_labeling

        self._precision, self._recall = 0, 0

    def eval(self):
        """
        Calculate the f1-score of produced hypotheses comparing with references
        Returns:
            score (float): evaluation score
        """
        if self._score is not None:
            return self._score
        else:
            if isinstance(self._target_label, int):
                self._precision, self._recall = self._fast_precision_recall()
            else:
                self._precision, self._recall = self._precision_recall()
            self._score = self._precision * self._recall * 2 / (self._precision + self.recall)
        return self._score

    def _precision_recall(self):
        true_positive, false_positive, true_negative, false_negative = 1e-8, 0, 0, 0
        
        if self._is_labeling:
            hypotoken, reftoken = [], []
            for hypo, ref in zip(self.hypos, self.refs):
                hypotoken.extend(hypo)
                reftoken.extend(ref)
        else:
            reftoken, hypotoken = self.refs, self.hypos
        reftoken = [1 if r >= 0.5 else 0 for r in reftoken]
        hypotoken = [1 if h >= 0.5 else 0 for h in hypotoken]
            
        for hypo, ref in zip(hypotoken, reftoken):
            if ref == self._target_label:
                if hypo == ref:
                    true_positive += 1
                else:
                    false_negative += 1
            else:
                if hypo == ref:
                    true_negative += 1
                else:
                    false_positive += 1
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        return precision, recall

    def _fast_precision_recall(self):
        import torch

        if self._is_labeling:
            hypotoken, reftoken = [], []
            for hypo, ref in zip(self.hypos, self.refs):
                hypotoken.extend(hypo)
                reftoken.extend(ref)
        else:
            reftoken, hypotoken = self.refs, self.hypos
        reftoken = [1 if r >= 0.5 else 0 for r in reftoken]
        hypotoken = [1 if h >= 0.5 else 0 for h in hypotoken]

        hypos = torch.LongTensor(hypotoken)
        refs = torch.LongTensor(reftoken)

        from paragen.utils.runtime import Environment
        env = Environment()
        if env.device.startswith('cuda'):
            hypos, refs = hypos.cuda(), refs.cuda()
        with torch.no_grad():
            true_mask = refs.eq(self._target_label)
            pos_mask = hypos.eq(self._target_label)
            true_positive = true_mask.masked_fill(~pos_mask, False).long().sum().data.item() + 1e-8
            false_positive = (~true_mask).masked_fill(~pos_mask, False).long().sum().data.item()
            false_negative = true_mask.masked_fill(pos_mask, False).long().sum().data.item()
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
        return precision, recall

    @property
    def precision(self):
        return self._precision

    @property
    def recall(self):
        return self._recall
