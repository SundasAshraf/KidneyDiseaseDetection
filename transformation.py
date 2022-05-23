def present_to_numeric(x):
    if x == 'present': return 2
    if x == 'notpresent':   return 1


def normal_to_numeric(x):
    if x == 'normal': return 2
    if x == 'abnormal':   return 1


def yes_to_numeric(x):
    if x == 'yes': return 2
    if x == 'no':   return 1


def good_to_numeric(x):
    if x == 'good': return 2
    if x == 'poor': return 1
def labels_to_numeric(x):
    if x == 'ckd': return 1
    if x == 'notckd': return 0