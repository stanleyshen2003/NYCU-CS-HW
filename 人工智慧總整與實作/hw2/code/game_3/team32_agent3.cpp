/*
teamID: 32
team name: 廢羊羊
group members:
	109350008 張詠哲
	110705013 沈昱宏
*/

#include "STcpClient.h"
#include <stdlib.h>
#include <iostream>
#include <vector>
#include "./mcts_yz.h"

MCTS_agent game1_agent;
/*
    選擇起始位置
    選擇範圍僅限場地邊緣(至少一個方向為牆)
    
    return: init_pos
    init_pos=<x,y>,代表你要選擇的起始位置
    
*/
std::vector<int> InitPos(int mapStat[12][12])
{
	std::vector<int> init_pos;
	init_pos.resize(2);

	/*
		Write your code here
	*/
	int ini_sheep_state[12][12] = {0};
	GameState ini_game_state(mapStat, ini_sheep_state, '2');
	Action ini_pos_action = game1_agent.decide_inipos(ini_game_state);

	init_pos[0] = ini_pos_action.x;
	init_pos[1] = ini_pos_action.y; 
    
    
    return init_pos;
}

/*
	產出指令
    
    input: 
	playerID: 你在此局遊戲中的角色(1~4)
    mapStat : 棋盤狀態, 為 12*12矩陣, 
					0=可移動區域, -1=障礙, 1~4為玩家1~4佔領區域
    sheepStat : 羊群分布狀態, 範圍在0~16, 為 12*12矩陣

    return Step
    Step : <x,y,m,dir> 
            x, y 表示要進行動作的座標 
            m = 要切割成第二群的羊群數量
            dir = 移動方向(1~9),對應方向如下圖所示
            1 2 3
			4 X 6
			7 8 9
*/
std::vector<int> GetStep(int playerID,int mapStat[12][12], int sheepStat[12][12])
{
	std::vector<int> step;
	step.resize(4);

	/*
		Write your code here
	*/
    char turn = playerID + '1';
	GameState gamestate(mapStat, sheepStat, turn);
	Action decided_action = game1_agent.decide_step(gamestate);

	step[0] = decided_action.x;
	step[1] = decided_action.y;
	step[2] = decided_action.n;
	step[3] = (decided_action.dir<4)?decided_action.dir+1:decided_action.dir+2;
    cout << "Step: " << step[0] << " " << step[1] << " " << step[2] << " " << step[3] << endl;
    return step;
}

int main()
{
	int id_package;
	int playerID;
    int mapStat[12][12];
    int sheepStat[12][12];

	// player initial
	GetMap(id_package,playerID,mapStat);
	std::vector<int> init_pos = InitPos(mapStat);
	SendInitPos(id_package,init_pos);

	while (true)
	{
		if (GetBoard(id_package, mapStat, sheepStat))
			break;
		// hide other player's sheep number start
		for (int i = 0; i < 12; i++)
			for (int j = 0; j < 12; j++)
				if (mapStat[i][j] != playerID)
					sheepStat[i][j] = 0;
		// hide other player's sheep number end 
		std::vector<int> step = GetStep(playerID,mapStat,sheepStat);
		SendStep(id_package, step);
	}
	// DON'T MODIFY ANYTHING IN THIS WHILE LOOP OR YOU WILL GET 0 POINT IN THIS QUESTION
}
