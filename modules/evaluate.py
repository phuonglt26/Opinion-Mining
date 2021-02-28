
def cal_aspect_prf(goldens, predicts, verbal=False):
    """

    :param verbal:
    :param num_of_aspect:
    :param list of models.AspectOutput goldens:
    :param list of models.AspectOutput predicts:
    :return:
    """
    tp = 0
    fp = 0
    fn = 0

    for g, p in zip(goldens, predicts):

        if g.scores == p.scores == 1:
            tp += 1
        elif g.scores == 1:
            fn += 1
        elif p.scores == 1:
            fp += 1

    p = tp/(tp+fp)
    r = tp/(tp+fn)
    f1 = 2*p*r/(p+r)

    # micro_p = sum(tp)/(sum(tp)+sum(fp))
    # micro_r = sum(tp)/(sum(tp)+sum(fn))
    # micro_f1 = 2*micro_p*micro_r/(micro_p+micro_r)
    #
    # macro_p = sum(p)/5
    # macro_r = sum(r)/5
    # macro_f1 = sum(f1)/5

    if verbal:
        print('p:', p)
        print('r:', r)
        print('f1:', f1)

    return p, r, f1,
