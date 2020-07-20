import os
import pickle


class PartOfSpeechTagging:
    """
    词性标注类

    Attribute:
        model_file: 用于记录存取算法中间结果，不用每次都训练模型
        state_list: 状态值结合，采用simultaneous思想的联合模型方法，将基于字标注的分词方法与词性标注结合起来，使用复核标注集
            ag: 形语素; a: 形容词; ad: 副形词; an: 名形词; b: 区别词; bg:区别语素; c: 连词; dg: 副语素; d: 副词; e: 叹词;
            f: 方位词; g: 语素; h: 前接成分; i: 成语; j: 简称略语; k: 后接成分; l: 习用语; m: 数词; mg:数语素; ng: 名语素; n: 名词;
            nr: 人名; ns: 地名; nt: 机构团体; nx: 字母专名; nz: 其他专名; o: 拟声词; p: 介词; q: 量词; r: 代词; s: 处所词;
            tg: 时语素; t: 时间词; u: 助词; vg: 动语素; v: 动词; vd: 副动词; vn: 名动词; w: 标点符号; x: 非语素字; y: 语气词;
            z: 状态词;
        load_para: 参数加载，用于判断是否需要重新加载model_file
        word_dic: 记录词语及其词性的字典
    """

    def __init__(self):
        self.model_file = './data/hmm_model.pkl'

        self.state_list = ['B_ag', 'B_a', 'B_ad', 'B_an', 'B_b', 'B_bg', 'B_c', 'B_dg', 'B_d', 'B_e', 'B_f',
                           'B_g', 'B_h', 'B_i', 'B_j', 'B_k', 'B_l', 'B_m', 'B_mg', 'B_ng', 'B_n', 'B_nr',
                           'B_ns', 'B_nt', 'B_nx', 'B_nz', 'B_o', 'B_p', 'B_q', 'B_r', 'B_s', 'B_tg', 'B_t',
                           'B_u', 'B_vg', 'B_v', 'B_vd', 'B_vn', 'B_w', 'B_x', 'B_y', 'B_z',
                           'M_ag', 'M_a', 'M_ad', 'M_an', 'M_b', 'M_bg', 'M_c', 'M_dg', 'M_d', 'M_e', 'M_f',
                           'M_g', 'M_h', 'M_i', 'M_j', 'M_k', 'M_l', 'M_m', 'M_mg',  'M_ng', 'M_n', 'M_nr',
                           'M_ns', 'M_nt', 'M_nx', 'M_nz', 'M_o', 'M_p', 'M_q', 'M_r', 'M_s', 'M_tg', 'M_t',
                           'M_u', 'M_vg', 'M_v', 'M_vd', 'M_vn', 'M_w', 'M_x', 'M_y', 'M_z',
                           'E_ag', 'E_a', 'E_ad', 'E_an', 'E_b', 'E_bg', 'E_c', 'E_dg', 'E_d', 'E_e', 'E_f',
                           'E_g', 'E_h', 'E_i', 'E_j', 'E_k', 'E_l', 'E_m', 'E_mg',  'E_ng', 'E_n', 'E_nr',
                           'E_ns', 'E_nt', 'E_nx', 'E_nz', 'E_o', 'E_p', 'E_q', 'E_r', 'E_s', 'E_tg', 'E_t',
                           'E_u', 'E_vg', 'E_v', 'E_vd', 'E_vn', 'E_w', 'E_x', 'E_y', 'E_z',
                           'S_ag', 'S_a', 'S_ad', 'S_an', 'S_b', 'S_bg', 'S_c', 'S_dg', 'S_d', 'S_e', 'S_f',
                           'S_g', 'S_h', 'S_i', 'S_j', 'S_k', 'S_l', 'S_m', 'S_mg',  'S_ng', 'S_n', 'S_nr',
                           'S_ns', 'S_nt', 'S_nx', 'S_nz', 'S_o', 'S_p', 'S_q', 'S_r', 'S_s', 'S_tg', 'S_t',
                           'S_u', 'S_vg', 'S_v', 'S_vd', 'S_vn', 'S_w', 'S_x', 'S_y', 'S_z'
                           ]

        self.load_para = False
        self.word_dic = {}

    def try_load_model(self, trained):
        """
        用于加载已计算的中间结果，当需要重新训练时，需初始化清空结果
        A_dic: 转移概率
        B_dic: 发射概率
        Pi_dic: 初始概率

        :param trained: 判断是否需要重新训练
        :return:
        """
        if trained:
            with open(self.model_file, 'rb') as f:
                self.A_dic = pickle.load(f)
                self.B_dic = pickle.load(f)
                self.Pi_dic = pickle.load(f)
                self.load_para = True
        else:
            self.A_dic = {}
            self.B_dic = {}
            self.Pi_dic = {}
            self.load_para = False

    def train(self, path):
        """
        计算转移概率、发射概率以及初始概率

        :param path:
        :return:
        """

        # 重置几个概率矩阵
        self.try_load_model(False)

        # 统计状态出现次数, 求P(o)
        count_dic = {}

        def init_parameters():
            """
            初始化参数

            :return:
            """
            for state in self.state_list:
                self.A_dic[state] = {s: 0.0 for s in self.state_list}
                self.B_dic[state] = {}
                self.Pi_dic[state] = 0.0
                count_dic[state] = 0

        def make_word_list(line):
            """
            针对每个句子，拆分出来词语与词性，将词语与词性组成的字典返回
            :param line: 输入的每个句子
            :return:
            """
            if line.find('[') != -1:
                # 如果这行文字中存在[]
                first_index = 0
                del_length = 0  # 中间删除的字符的长度
                for i, w in enumerate(line):
                    if w == '[':
                        # 记录出现[的下标
                        first_index = i - del_length
                    if w == ']':
                        # 记录出现]的下标，因为只有]出现了才证明两个匹配了，可以进行后面的内容
                        second_index = i - del_length
                        inner_line = line[first_index: second_index + 3]
                        inner_word = inner_line.replace('[', '')  # 删除[
                        inner_word = inner_word.replace(']', '/')  # 将]替换为/，方便后续切割
                        inner_word = inner_word.split(' ')
                        whole_word = ''
                        for iw in inner_word:
                            # 将所有[]内的汉字拼接在一起成为长词
                            whole_word += iw.split('/')[0]

                        self.word_dic[whole_word] = line[second_index + 1: second_index + 3]  # 词性也添加至字典

                        line = line[: first_index] + line[second_index + 3:]

                        # 因为line是变化的，枚举长度是不变的，就会造成每合并一次之后line的下标和i不匹配，
                        # 所以需要把减少的这一部分给去除，因为i不变，而减少的长度是越来越长，所以需要累加
                        # 每次减少的长度是两个括号+末尾词性的长度
                        del_length += second_index + 3 - first_index

            word_list = line.split(' ')
            for word in word_list:
                # 字典的键为词，值为词性
                if word == '':
                    # 切割后有空字符串，需要跳过
                    continue

                self.word_dic[word.split('/')[0]] = word.split('/')[1]

        def make_label(word, tagging):
            """
            针对每个词语制作标签
            :param word: 输入的词语
            :param tagging: 该词语的词性
            :return:
            """
            out_text = []
            tagging = tagging.lower()
            if tagging == 'm':
                # 如果这个词语是数字，那么单独成词
                out_text.append('S_' + tagging)
            elif len(word) == 1:
                # 如果这个词语长度为1，且不为数字，那么单独成词
                out_text.append('S_' + tagging)
            else:
                # 如果这个词语长度大于1，那么针对每个字设置相应的标签
                out_text += ['B_' + tagging] + ['M_' + tagging] * (len(word) - 2) + ['E_' + tagging]

            for t in out_text:
                if t not in self.state_list:
                    print(word)
                    print(t)

            return out_text

        init_parameters()
        line_num = -1

        words = set()
        with open(path, encoding='utf8') as f:
            for line in f:
                line_num += 1
                line = line.strip()

                if not line:
                    # 如果句子为空则跳过这轮循环
                    continue

                make_word_list(line)

                words |= set(self.word_dic.keys())

                line_state = []

                for w, t in self.word_dic.items():
                    # w: 每个词语, t: 每个词语的词性
                    line_state.extend(make_label(w, t))

                # for k, v in enumerate(line_state):
                #     count_dic[v] += 1
                #     if k == 0:
                #         # 计算初始概率
                #         self.Pi_dic[v] += 1
                #     else:
                #         # 计算转移概率与发射概率
                #         self.A_dic[line_state[k - 1]][v] += 1
                #         # self.B_dic[line_state[k]][]

    def viterbi(self):
        pass

    def cut(self):
        pass


if __name__ == '__main__':
    post = PartOfSpeechTagging()
    post.train("./data/people-daily.txt")

    for k, v in post.word_dic.items():
        if v not in post.state_list:
            print(k, v)
