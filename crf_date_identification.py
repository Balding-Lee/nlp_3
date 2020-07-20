import re
from datetime import datetime, timedelta
from dateutil.parser import parse
import jieba.posseg as psg

# 优化点:
#   1. check_time_valid(word)中的第一个if语句感觉可以优化的更简单明了一些
#   2. 核心正则表达式感觉还有些问题，普适性不强
#   3. 对于parse_datetime(msg)中传入year2dig和cn2dig的参数，感觉不太需要把最后一个字符去掉，有可能会造成错误，
#      比如: "三点十五"，那么就会去掉"五"，造成错误

# 预定义模板, 将具体文本转换成数字
UTIL_CN_NUM = {
    '零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
}
UTIL_CN_UTIL = {'十': 10, '百': 100, '千': 1000, '万': 10000}


def check_time_valid(word):
    """
    对拼接字符串近一步处理，以进行有效性判断
    :param word: time_res中的每一项(每一项切割出来的时间)
    :return: 
    """
    # match()匹配成功返回对象，否则返回None，
    # match是全匹配，即从头到尾，而$是匹配最后，从match源码来看，如果str是存在非数字的情况会直接返回None
    # 这里的意思就是清洗掉长度小于等于6的纯数字(小于等于6的意思是指非准确日期，比如2020)
    m = re.match('\d+$', word)
    if m:
        # 当正则表达式匹配成功时，判断句子的长度是否小于等于6，如果小于等于6，则返回None
        if len(word) <= 6:
            return None
    # 将"号"和"日"替换为"日",个人理解，这里是号和日后面莫名其妙跟了一串数字的情况
    word_1 = re.sub('[号|日]\d+$', '日', word)
    if word_1 != word:
        # 如果清洗出来的句子与原句子不同，则递归调用
        return check_time_valid(word_1)
    else:
        # 如果清洗出来的句子与原句子相同，则返回任意一个句子
        return word_1


def year2dig(year):
    """
    解析年份这个维度，主要是将中文或者阿拉伯数字统一转换为阿拉伯数字的年份
    :param year: 传入的年份(从列表的头到倒数第二个字，即假设有"年"这个字，则清洗掉"年")
    :return: 所表达的年份的阿拉伯数字或者None
    """
    res = ''
    for item in year:
        # 循环遍历这个年份的每一个字符
        if item in UTIL_CN_NUM.keys():
            # 如果这个字在UTIL_CN_NUM中，则转换为相应的阿拉伯数字
            res = res + str(UTIL_CN_NUM[item])
        else:
            # 否则直接相加
            # 这里已经是经历了多方面清洗后的结果了，基本到这里不会在item中出现异常的字符
            res = res + item

    m = re.match("\d+", res)
    if m:
        # 当m开头为数字时，执行下面操作，否则返回None
        if len(m.group(0)) == 2:
            # 这里是假设输入的话为"我要住到21年..."之类的，那么year就只有2个字符，即这里m == 21，
            # 那么就通过当前年份除100的整数部分再乘100最后加上这个数字获得最终年份
            # 即int(2020 / 100) * 100 + int("21")
            return int(datetime.today().year / 100) * 100 + int(m.group(0))
        else:
            # 否则直接返回该年份
            return int(m.group(0))
    else:
        return None


def cn2dig(src):
    """
    除了年份之外的其余时间的解析
    :param src: 除了年份的其余时间(从列表的头到倒数第二个字，即假设有"月"这个字，则清洗掉"月")
    :return rsl: 返回相应的除了年份的其余时间的阿拉伯数字
    """
    if src == "":
        # 如果src为空，那么直接返回None，又进行一次清洗
        return None
    m = re.match("\d+", src)
    if m:
        # 如果m是数字则直接返回该数字
        return int(m.group(0))
    rsl = 0
    unit = 1
    for item in src[:: -1]:
        # 从后向前遍历src
        if item in UTIL_CN_UTIL.keys():
            # 如果item在UTIL_CN_UTIL中，则unit为这个字转换过来的阿拉伯数字
            # 即假设src为"三十"，那么第一个item为"十"，对应的unit为10
            unit = UTIL_CN_UTIL[item]
        elif item in UTIL_CN_NUM.keys():
            # 如果item不在UTIL_CN_UTIL而在UTIL_CN_NUM中，则转换为相应的阿拉伯数字并且与unit相乘
            # 就假设刚刚那个"三十"，第二个字为"三"，对应的num为3，rsl就为30
            num = UTIL_CN_NUM[item]
            rsl += num * unit
        else:
            # 如果都不在，那么就不是数字，就直接返回None
            return None

    if rsl < unit:
        # 如果出现"十五"这种情况，那么是先执行上面的elif，即rsl = 5，再执行if，即unit = 10，
        # 这时候rsl < unit，那么执行相加操作
        rsl += unit

    return rsl


def parse_datetime(msg):
    """
    将每个提取到的文本日期串进行时间转换
    
    实现方式:
        通过正则表达式将日期串进行切割，分成'年''月''日''时''分''秒'等具体维度，然后针对每个子维度单独再进行识别
    :param msg: 初步清洗后的每一个有关时间的句子
    :return: 如果时间可以通过parse解析，那么返回解析后的时间
             如果不能够解析，那么返回自行处理后的时间
             否则返回None
    """
    if msg is None or len(msg) == 0:
        # 如果之前清洗失误或者其他原因造成的句子为空，则返回None
        return None

    try:
        # 将日期格式化成datetime的时间，fuzzy=True: 允许时间是模糊时间，如:
        # Today is January 1, 2047 at 8:21:00AM
        # dt = parse(msg, fuzzy=True)
        dt = parse(msg)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        m = re.match(r"([0-9零一二两三四五六七八九十]+ 年)? ([0-9一二两三四五六七八九十]+ 月)? "
                     r"([0-9一二两三四五六七八九十]+ [号日])? ([上中下午晚早]+)?"
                     r"([0-9零一二两三四五六七八九十百]+[点:.时])?([0-9零一二三四五六七八九十百]+ 分?)?"
                     r"([0-9零一二三四五六七八九十百]+ 秒)?", msg)
        if m.group(0) is not None:
            res = {
                'year': m.group(1),
                "month": m.group(2),
                "day": m.group(3),
                "hour": m.group(5) if m.group(5) is not None else '00',
                "minute": m.group(6) if m.group(6) is not None else '00',
                "second": m.group(7) if m.group(7) is not None else '00',
            }
            params = {}

            for name in res:
                if res[name] is not None and len(res[name]) != 0:
                    if name == 'year':
                        # 如果是年份，tmp就进入year2dig
                        tmp = year2dig(res[name][: -1])
                    else:
                        # 否则就是其他时间，那么进入cn2dig
                        tmp = cn2dig(res[name][: -1])
                    if tmp is not None:
                        # 当tmp之中存在阿拉伯数字的时候，params就为该tmp
                        params[name] = int(tmp)
            # 使用今天的时间格式，然后将数字全部替换为params[]中的内容
            target_date = datetime.today().replace(**params)
            is_pm = m.group(4)
            if is_pm is not None:
                # 如果文字中有"中午"、"下午"、"晚上"二字
                if is_pm == u'下午' or is_pm == u'晚上' or is_pm == u'中午':
                    hour = target_date.time().hour  # 获取刚刚解析的时间的小时
                    if hour < 12:
                        # 如果小时小于12，那么替换为24小时制
                        target_date = target_date.replace(hour=hour + 12)
            return target_date.strftime("%Y-%m-%d %H:%M:%S")
        else:
            return None


def time_extract(text):
    """
    思路:
        通过jieba分词将带有时间信息的词进行切分，记录连续时间信息的词。
        使用了词性标注，提取"m(数字)"和"t(时间)"词性的词。
    
    规则约束:
        对句子进行解析，提取其中所有能表示日期时间的词，并进行上下文拼接

    :param text: 每一个请求文本
    :return: 
    """
    print("--------------------")

    time_res = []
    word = ''
    key_date = {'今天': 0, '明天': 1, '后天': 2}
    for k, v in psg.cut(text):
        # k: 词语, v: 词性
        if k in key_date:
            # 当k存在于key_date中时
            if word != '':
                # 如果word不为空时, 列表中添加相应的词语
                time_res.append(word)
            # 获取系统当前时间，并且获取句子中时间的跨度(0, 1, 2)，通过当前时间 + 时间跨度获得几天后的时间
            word = (datetime.today() + timedelta(days=key_date.get(k, 0))) \
                .strftime('%Y {0} %m {1} %d {2} ').format('年', '月', '日')
        elif word != '':
            # 如果k不存在于key_date时，word不为空
            if v in ['m', 't']:
                # 当词性为数字或时间时，添加至word中
                word = word + k
            else:
                # 当词性不为数字或时间时，将word放入time_res，同时清空word
                time_res.append(word)
                word = ''
        elif v in ['m', 't']:
            # 当k不存在于key_date中，且word为空时，如果词性是数字或时间时，word为该词语
            word = k
    if word != '':
        # word中可能存放的值:
        #   1. 通过词性标注后获得的时间跨度后的时间
        #   2. 非key_date中的时间或数字
        # 即只有k不存在于key_date，word不为空，词性不为数字或时间时，word才为空，进入不了这个if语句
        time_res.append(word)

    # 如果返回的结果是None，则直接清洗，否则放入集合中
    result = list(filter(lambda x: x is not None, [check_time_valid(w) for w in time_res]))
    final_res = [parse_datetime(w) for w in result]

    return [x for x in final_res if x is not None]


if __name__ == '__main__':
    text1 = '我要住到明天下午三点'
    print(text1, time_extract(text1), sep=':')

    text2 = '预定28号的房间'
    print(text2, time_extract(text2), sep=':')

    text3 = '我要从26号下午4点住到8月2号'
    print(text3, time_extract(text3), sep=':')

    text4 = '我要预定今天到30号的房间'
    print(text4, time_extract(text4), sep=':')

    text5 = '今天30号呵呵'
    print(text5, time_extract(text5), sep=':')



