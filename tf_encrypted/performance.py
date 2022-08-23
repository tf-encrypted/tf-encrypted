import time


class Performance:

    timeSentinel = []

    @staticmethod
    def time_log(tag):
        stack = Performance.timeSentinel
        if len(stack) == 0 or not stack[-1][0] == tag:
            stack.append((tag, time.time()))
            print("[Time] " + tag + " start: " + str(time.strftime("%Y%m%d-%H:%M:%S")))
        else:
            e = stack.pop()
            print("[Time] " + tag + ": " + str(time.time() - e[1]) + " s")
