'''
    Copyright © 2019 by Phillip Chang
'''
import struct
import socket
from numpy import zeros, array

socketServer = None
infoServer = ["localhost", 8887]
'''
    *   請將 idTeam 改成組別    *
'''
idTeam = -1


def _Connect(ip, port):
    socketCurrent = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    addrServer = (ip, port)
    error = socketCurrent.connect_ex(addrServer)
    if (error == 0):
        print("Connect to server")
        return socketCurrent

    socketCurrent.close()
    return None


def _RecvUntil(socketRecv, cntByte):
    if (socketRecv is None):
        return None
    try:
        rbData = socketRecv.recv(cntByte)
    except socket.error as _:
        return None

    if (len(rbData) != cntByte):
        return None

    return rbData


def _SendAll(socketSend, rbData):
    if (socketSend is None):
        return False
    try:
        resultSend = socketSend.sendall(rbData)
    except socket.error as e:
        print(e)
        return False

    return resultSend is None


def _ConnectToServer(cntRecursive=0):
    global socketServer
    global infoServer
    global idTeam
    global is_connect

    if (cntRecursive > 3):
        print("[Error] : maximum connection try reached!")
        return
    while (socketServer is None):
        socketServer = _Connect(infoServer[0], infoServer[1])

    structHeader = struct.Struct("i")
    rbHeader = structHeader.pack(idTeam)
    if (not _SendAll(socketServer, rbHeader)):
        socketServer.close()
        socketServer = None
        _ConnectToServer(cntRecursive + 1)


def _ReconnectToServer():
    global socketServer

    if (socketServer is not None):
        socketServer.close()
        socketServer = None

    _ConnectToServer()

def _StopConnect():
    global socketServer

    if (socketServer is not None):
        socketServer.close()
        socketServer = None


'''
    取得初始化地圖
'''
def GetMap():
    global socketServer

    if socketServer is None:
        _ConnectToServer()
        if socketServer is None:
            return 0, None, None

    # recv
    structHeader = struct.Struct("ii")
    structItem = struct.Struct("i")

    rbHeader = _RecvUntil(socketServer, structHeader.size)
    if rbHeader is None:
        print("[Error] : connection lose, trying to reconnect...")
        socketServer.close()
        socketServer = None
        return GetMap()

    (codeHeader, id_package) = structHeader.unpack(rbHeader)

    # unpack playerID(1~4)
    rbPlayer = _RecvUntil(socketServer, structItem.size)
    if rbPlayer is None:
        print("[Error] : connection lose, trying to reconnect...")
        socketServer.close()
        socketServer = None
        return GetMap()
    playerID = structItem.unpack(rbPlayer)[0]

    # unpack map
    map_current = zeros([12, 12])
    for i in range(12):
        temp = []
        for _ in range(12):
            rbBoard = _RecvUntil(socketServer, structItem.size)
            if rbBoard is None:
                print("[Error] : connection lose, trying to reconnect...")
                socketServer.close()
                socketServer = None
                return GetMap()
            itemBoard = structItem.unpack(rbBoard)[0]
            temp.append(itemBoard)
        map_current[i] = array(temp)

    return id_package, playerID,map_current



'''
    取得當前遊戲狀態

    return (stop_program, id_package, board, is_black)
    stop_program : True 表示當前應立即結束程式，False 表示當前輪到自己下棋
    id_package : 當前棋盤狀態的 id，回傳移動訊息時需要使用
    mapStat: 當前棋盤佔領的狀態
    gameStat 當前羊群分布狀態
'''
def GetBoard():
    global socketServer

    if socketServer is None:
        _ConnectToServer()
        if socketServer is None:
            return True, 0, None, None

    # recv
    structHeader = struct.Struct("ii")
    structItem = struct.Struct("i")

    rbHeader = _RecvUntil(socketServer, structHeader.size)
    if rbHeader is None:
        print("[Error] : connection lose, trying to reconnect...")
        socketServer.close()
        socketServer = None
        return GetBoard()

    (codeHeader, id_package) = structHeader.unpack(rbHeader)
    if codeHeader == 0:
        return True, 0, None, None

    # unpack map
    map_current = zeros([12, 12])
    for i in range(12):
        temp = []
        for _ in range(12):
            rbBoard = _RecvUntil(socketServer, structItem.size)
            if rbBoard is None:
                print("[Error] : connection lose, trying to reconnect...")
                socketServer.close()
                socketServer = None
                return GetBoard()
            itemBoard = structItem.unpack(rbBoard)[0]
            temp.append(itemBoard)
        map_current[i] = array(temp)

    # unpack sheep condition
    game_current = zeros([12, 12])
    for i in range(12):
        temp = []
        for _ in range(12):
            rbSheep = _RecvUntil(socketServer, structItem.size)
            if rbSheep is None:
                print("[Error] : connection lose, trying to reconnect...")
                socketServer.close()
                socketServer = None
                return GetBoard()
            itemBoard = structItem.unpack(rbSheep)[0]
            temp.append(itemBoard)
        game_current[i] = array(temp)

    return False, id_package, map_current, game_current


'''
    傳送起始位置座標, pos=[x,y]
'''
def SendInitPos(id_package,pos):
    global socketServer

    if (socketServer is None):
        print("[Error] : trying to send step before connection is established")
        return

    # send
    structItem = struct.Struct("ii")
    rbHeader = struct.pack("ii", 1, id_package)
    rbHeader += structItem.pack(pos[0],pos[1])

    # retry once
    if not _SendAll(socketServer, rbHeader):
        print("[Error] : connection lose, trying to reconnect")
        _ReconnectToServer()


'''
    向 server 傳達移動訊息
    id_package : 想要回復的訊息的 id_package
    Step = [(x, y),m,dir]
            x, y 表示要進行動作的座標 
            m = 要切割成第二群的羊群數量
            dir = 移動方向(1~9),對應方向如下圖所示
            1 2 3
			4 X 6
			7 8 9

    return 函數是否執行成功
'''

def SendStep(id_package, Step):
    global socketServer

    if (socketServer is None):
        print("[Error] : trying to send step before connection is established")
        return

    # send

    structItem = struct.Struct("iiii")
    rbHeader = struct.pack("ii", 1, id_package)
    rbHeader += structItem.pack(Step[0][0], Step[0][1], Step[1], Step[2])

    # retry once
    if not _SendAll(socketServer, rbHeader):
        print("[Error] : connection lose, trying to reconnect")
        _ReconnectToServer()
