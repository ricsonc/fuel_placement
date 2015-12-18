
def rle(lst):
    '''makes a run length encoding of input list
    useful for analyzing fuel_placement_tests outputs'''
    runs = []
    lens = []
    lastitem = lst.pop(0)
    curlen = 1
    for item in lst:
        if item == lastitem:
            curlen += 1
        else:
            runs.append(lastitem)
            lens.append(curlen)
            curlen = 1
            lastitem = item
    runs.append(lastitem)
    lens.append(curlen)
    return zip(runs, lens)
        
