import random
import string


seed = "1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def  getrandomname():
    sa = []
    for k in range(8):
        sa.append(random.choice(seed))
    return ''.join(sa)