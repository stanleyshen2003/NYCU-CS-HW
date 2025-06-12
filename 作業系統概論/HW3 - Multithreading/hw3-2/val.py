import argparse


def make_parser():
    parser = argparse.ArgumentParser("hw3-n val")
    parser.add_argument("testcase")
    parser.add_argument("answer_file")
    return parser


def main(args):
    testcase = args.testcase
    answer_file = args.answer_file

    vertex_num = None
    adjacent_matrix = None
    with open(f"testcase/case{testcase}.txt", "r") as graph:
        vertex_num, _ = map(int, graph.readline().split(" "))
        adjacent_matrix = [[] for _ in range(vertex_num)]

        for line in graph.readlines():
            u, v = map(int, line.split(" "))
            adjacent_matrix[u].append(v)
            adjacent_matrix[v].append(u)

    V = set()
    with open(answer_file, "r") as MID:
        for v in map(int, MID.read().split()):
            if v in V:
                exit(1)
            V.add(v)
            for u in adjacent_matrix[v]:
                V.add(u)

    if vertex_num != len(V):
        exit(1)

    return

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
