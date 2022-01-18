try:
    from lightseq.training.ops.pytorch.adam import LSAdam as Adam

    from paragen.optim import register_optim


    @register_optim
    class LSAdam(Adam):

        pass

except Exception:
    pass
