class RnnLayer:
    def __init__(self, inputSize:int, hiddenSize:int):
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.w = 