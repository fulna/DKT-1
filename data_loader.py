import numpy as np
import math


class DATA(object):
    def __init__(self, n_question, seqlen, separate_char, name="data"):
        # In the ASSISTments2009 dataset:
        # param: n_queation = 110
        #        seqlen = 200
        self.separate_char = separate_char
        self.n_question = n_question
        """
        self.seqlen = seqlen+1
        """
        self.seqlen = seqlen

    ### data format
    ### 15
    ### 1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
    ### 0,1,1,1,1,1,0,0,1,1,1,1,1,0,0
    def load_data(self, path):
        f_data = open(path, 'r')
        qa_inputdata = []
        q_target_data = []
        qa_data = []
        for lineID, line in enumerate(f_data):
            line = line.strip()
            # lineID starts from 0
            if lineID % 3 == 1:
                Q = line.split(self.separate_char)
                if len(Q[len(Q) - 1]) == 0:
                    Q = Q[:-1]
                # print(len(Q))
            elif lineID % 3 == 2:
                A = line.split(self.separate_char)
                if len(A[len(A) - 1]) == 0:
                    A = A[:-1]
                # print(len(A),A)

                # start split the data
                n_split = 1
                # print('len(Q):',len(Q))
                seq_len = self.seqlen+1 #现在是取的801个
                if len(Q) > seq_len:
                    n_split = math.floor(len(Q) / seq_len)
                    if len(Q) % seq_len:
                        n_split = n_split + 1
                # print('n_split:',n_split)
                for k in range(n_split):
                    question_sequence = []
                    answer_sequence = []
                    if k == n_split - 1:
                        end_index = len(A)
                    else:
                        end_index = (k + 1) * seq_len
                    for i in range(k * seq_len, end_index):
                        if len(Q[i]) > 0:
                            # int(A[i]) is in {0,1}
                            x_index = int(Q[i]) + int(A[i]) * self.n_question
                            question_sequence.append(int(Q[i]))
                            answer_sequence.append(x_index)
                        else:
                            print(Q[i])
                    # print('instance:-->', len(instance),instance)
                    if len(question_sequence) > 1:
                        inputx_pair = answer_sequence[:-1]
                        question_target_sequence = question_sequence[1:]
                        answer_sequence = answer_sequence[1:]
                        qa_inputdata.append(inputx_pair)
                        q_target_data.append(question_target_sequence)
                        qa_data.append(answer_sequence)

        f_data.close()
        ### data: [[],[],[],...] <-- set_max_seqlen is used
        ### convert data into ndarrays for better speed during training
        q_inputdataArray = np.zeros((len(qa_inputdata), self.seqlen))
        q_target_dataArray = np.zeros((len(qa_inputdata), self.seqlen))
        for j in range(len(qa_inputdata)):
            dat = qa_inputdata[j]
            qt_dat = q_target_data[j]
            q_inputdataArray[j, :len(dat)] = dat
            q_target_dataArray[j, :len(qt_dat)] = qt_dat

        qa_dataArray = np.zeros((len(qa_data), self.seqlen))
        for j in range(len(qa_data)):
            dat = qa_data[j]
            qa_dataArray[j, :len(dat)] = dat
        # dataArray: [ array([[],[],..])] Shape: (3633, 200)
        return q_inputdataArray,q_target_dataArray, qa_dataArray

    def generate_all_index_data(self, batch_size):
        n_question = self.n_question
        batch = math.floor(n_question / self.seqlen)
        if self.n_question % self.seqlen:
            batch += 1

        seq = np.arange(1, self.seqlen * batch + 1)
        zero_index = np.arange(n_question, self.seqlen * batch)
        zero_index = zero_index.tolist()
        seq[zero_index] = 0
        q = seq.reshape((batch, self.seqlen))
        q_dataArray = np.zeros((batch_size, self.seqlen))
        q_dataArray[0:batch, :] = q
        return q_dataArray

