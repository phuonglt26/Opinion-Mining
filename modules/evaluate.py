def cal_sentiment_prf(f1, num_of_aspect, verbal=False):
    f1_score = '{}, {}, {}, {}, {}'.format(f1[0], f1[1], f1[2], f1[3], f1[4])
    macro = sum(f1) / num_of_aspect
    title = 'price,service,safety,quality,ship,authenticity'
    title = 'staff service, room standard, food, location price, facilities'
    if verbal:
        print(title)
        print(f1_score)
        print(macro)


    # output = _p + ', ' + micro_p + ', ' + macro_p + '\n' + _r + '\n' + _f1
    output = ''
    outputs = title + output
    return outputs

# return p, r, f1, (micro_p, micro_r, micro_f1), (macro_p, macro_r, macro_f1)


# def cal_sentiment_prf(tp, fp, fn, p, r, f1, num_of_aspect, verbal=False):
#     micro_p = sum(tp) / (sum(tp) + sum(fp))
#     micro_r = sum(tp) / (sum(tp) + sum(fn))
#     micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
#
#     macro_p = sum(p) / (2 * num_of_aspect)
#     macro_r = sum(r) / (2 * num_of_aspect)
#     macro_f1 = sum(f1) / (2 * num_of_aspect)
#
#     _micro_p = 'micro_p, {}'.format(micro_p)
#     _micro_r = 'micro_r, {}'.format(micro_r)
#     _micro_f1 = 'micro_f1, {}'.format(micro_f1)
#     _macro_p = 'macro_p, {}'.format(macro_p)
#     _macro_r = 'macro_r, {}'.format(macro_r)
#     _macro_f1 = 'macro_f1, {}'.format(macro_f1)
#
#     # tính cho negative
#     n = len(p)
#     _tp = [tp[i] for i in range(n) if i % 2 != 0]
#     _fp = [fp[i] for i in range(n) if i % 2 != 0]
#     _fn = [fn[i] for i in range(n) if i % 2 != 0]
#     _p = [p[i] for i in range(n) if i % 2 != 0]
#     _r = [r[i] for i in range(n) if i % 2 != 0]
#     _f1 = [f1[i] for i in range(n) if i % 2 != 0]
#     print(_p)
#     micro_p_neg = sum(_tp) / (sum(_tp) + sum(_fp))
#     micro_r_neg = sum(_tp) / (sum(_tp) + sum(_fn))
#     micro_f1_neg = 2 * micro_p_neg * micro_r_neg / (micro_p_neg + micro_r_neg)
#
#     macro_p_neg = sum(_p) / num_of_aspect
#     macro_r_neg = sum(_r) / num_of_aspect
#     macro_f1_neg = sum(_f1) / num_of_aspect
#
#     _micro_p_neg = 'micro_p, {}'.format(micro_p_neg)
#     _micro_r_neg = 'micro_r, {}'.format(micro_r_neg)
#     _micro_f1_neg = 'micro_f1, {}'.format(micro_f1_neg)
#     _macro_p_neg = 'macro_p, {}'.format(macro_p_neg)
#     _macro_r_neg = 'macro_r, {}'.format(macro_r_neg)
#     _macro_f1_neg = 'macro_f1, {}'.format(macro_f1_neg)
#
#     if verbal:
#         print(_micro_p)
#         print(_micro_r)
#         print(_micro_f1)
#         print(_macro_p)
#         print(_macro_r)
#         print(_macro_f1)
#         print('tiêu cực')
#         print(_micro_p_neg)
#         print(_micro_r_neg)
#         print(_micro_f1_neg)
#         print(_macro_p_neg)
#         print(_macro_r_neg)
#         print(_macro_f1_neg)
#
#     title = 'price,service,safety,quality,ship,authenticity, micro, macro'
#     # output = _p + ', ' + micro_p + ', ' + macro_p + '\n' + _r + '\n' + _f1
#     output = ''
#     outputs = title + output
#     return outputs
#
# # return p, r, f1, (micro_p, micro_r, micro_f1), (macro_p, macro_r, macro_f1)
