#include "Dungeon.h"
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <unistd.h>
Dungeon::Dungeon() {

}

void Dungeon::createPlayer() {
	cout << "please type your name : ";
	string name;
	cin >> name;
	this->player = Player(name, 100, 10, 10);
	this->player.setCurrentRoom(&rooms[0]);
	this->player.setPreviousRoom(&rooms[0]);
	//rooms[0].getObjects()[1]->triggerEvent(&player);
}

void Dungeon::createMap() {
	rooms = vector<Room>(12);
	for (int i = 0;i < 12;i++) {
		rooms[i].setIndex(i);
	}
	for (int i = 0;i < 12;i++) {
		rooms[i].setIsExit(false);
	}
	rooms[10].setIsExit(true);
	rooms[0].setRightRoom(&rooms[1]);
	rooms[1].setAllRoom(&rooms[0], &rooms[3], &rooms[2], &rooms[11]);
	rooms[2].setLeftRoom(&rooms[1]);
	rooms[3].setAllRoom(NULL, &rooms[4], NULL, &rooms[1]);
	rooms[4].setAllRoom(NULL, &rooms[7], &rooms[5], &rooms[3]);
	rooms[5].setAllRoom(&rooms[4], NULL, &rooms[6], NULL);
	rooms[6].setLeftRoom(&rooms[5]);
	rooms[7].setAllRoom(&rooms[10], &rooms[8], NULL, &rooms[4]);
	rooms[8].setAllRoom(NULL, &rooms[9], NULL, &rooms[7]);
	rooms[9].setDownRoom(&rooms[8]);
	rooms[10].setAllRoom(NULL, NULL, &rooms[7], NULL);
	rooms[11].setUpRoom(&rooms[1]);

	//add objects
	std::srand(std::time(nullptr));
	int num1 = rand() % 20 + 40;
	int num2 = rand() % 2 + 4;
	Object* npc = new NPC("god","god : hello!, welcome to the game! Enjoy your adventure!\ngod : Here is 10 dollar",{});
	Object* item = new Item("money",0,0,0,10);
	vector<Object*> obj = {npc,item};
	rooms[0].setObjects(obj);
	//cout<<&obj[1]<<"in obj[1]"<<endl;
	//cout<<&rooms[0].getObjects()[1]<<" in room"<<endl;
	//obj[1]->triggerEvent(&item);
    //rooms[0].getObjects()[1]->triggerEvent(&item);

    Object* item1 = new Item("unknown healthy potion",num1,num2,num2,0);
    Object* item2 = new Item("one-time recover potion",0,0,0,0);
	vector<Object*> obj2 = { item1,item2 };
	rooms[1].setObjects(obj2);

	Object* monster = new Monster("goblin","smallMonster","goblin: MONEY!!!\n",0,0,0);
	Object* item3 = new Item("chest",0,0,0,0);
	Object* item4 = new Item("small sword",0,15,0,0);
	Object* item5 = new Item("money",0,0,0,15);
	Object* item6 = new Item("revive armor",0,0,0,0);
	vector<Object*> obj3 = { monster, item3, item4,item5,item6 };
	rooms[2].setObjects(obj3);

	Item item20("time-sensitive prediction",0,0,0,3),item21("body strengthening",100,30,30,30);
	Object* npc1 = new NPC("fortune teller","fortune teller : Do you know what's going to happen...?",{item20,item21});
	vector<Object*> obj4 = { npc1 };
	rooms[3].setObjects(obj4);

	Object* monster1 = new Monster("vampire Zaff","smallMonster","vampire Zaff: I waaannnt BLLOOOOD!\n",0,0,0);
	Object* item7 = new Item("chest",0,0,0,0);
	Object* item8 = new Item("money",0,0,0,15);
	vector<Object*> obj5 = { monster1,item7, item8 };
	rooms[11].setObjects(obj5);

	num1 = rand() % 100 + 250;
	num2 = rand() % 10 + 35;
	int num3 = rand() % 10 + 35;
	Object* monster2 = new Monster("Brarzamuth the Devil","smallDevil","Brarzamuth the Devil: ARE YOU AFRAID??\n",0,0,0);
	Object* item9 = new Item("chest",0,0,0,0),*item10 = new Item("one-time recover potion",0,0,0,0),*item11 = new Item("money",0,0,0,50),*item12 = new Item("devil fruit - dark",num1,num2,num3,0),*item13 = new Item("Devil's egg",200,30,30,0);
	vector<Object*> obj6 = { monster2 ,item9,item10,item11, item12, item13 };
	rooms[4].setObjects(obj6);

	Object* npc2 = new NPC("Locked wizard","Locked Wizard : Only the one with demon powers can fight with the guardians...",{});
	vector<Object*> obj7 = { npc2 };
	rooms[7].setObjects(obj7);

	Object* monster3 = new Monster("Devil guardian","guardian","Devil guardian: protect our kingdom!\n", 700,140,140);
	Object* item14 = new Item("chest", 0, 0, 0, 0),*item15 = new Item("money", 0, 0, 0, 170);
	vector<Object*> obj8 = { monster3 ,item14, item15 };
	rooms[5].setObjects(obj8);

	Item item22("PURIFYING LASERGUN", 0, 0, 0, 200);
	Object* npc3 = new NPC("Laserman", "Laserman : beat the demons!!!!", { item22 });
	vector<Object*> obj9 = { npc3 };
	rooms[6].setObjects(obj9);

	num1 = rand() % 100 + 250;
	num2 = rand() % 10 + 35;
	num3 = rand() % 10 + 35;
	Object* monster4 = new Monster("Xuzgulun the Devil","smallDevil","Xuzgulun the Devil: ready to fight the DEVILLLL?\n",200,40,40);
	Object* item16 = new Item("chest",0,0,0,0),*item17 = new Item("one-time recover potion",0,0,0,0),*item18 = new Item("money",0,0,0,50),*item19 = new Item("devil fruit - vibrate",num1,num2,num3,0);
	vector<Object*> obj10 = { monster4,item16,item17,item18, item19 };
	rooms[8].setObjects(obj10);

	Item item23("unknown magic paper", 0, 0, 0, 70);
	Object* npc4 = new NPC("Black beard", "Black Beard : HAHAHAHAHA!!!!", { item23 });
	vector<Object*> obj11 = { npc4 };
	rooms[9].setObjects(obj11);

	Object* monster5 = new Monster("King of Devil guardian","boss","King of Devil guardian: I'm the chosen one\n", 999999, 99999, 99999);
	vector<Object*> obj12 = { monster5 };
	rooms[10].setObjects(obj12);
	//rooms[0].getObjects()[1]->triggerEvent(item);
}

void Dungeon::handleMovement() {
	string input = "";
	Room* allRoom[4] = {};
	string ways[] = { "a","w","d","s" };
	allRoom[0] = player.getCurrentRoom()->getLeftRoom();
	allRoom[1] = player.getCurrentRoom()->getUpRoom();
	allRoom[2] = player.getCurrentRoom()->getRightRoom();
	allRoom[3] = player.getCurrentRoom()->getDownRoom();
	bool finish = false;
	cout << "\nchoose your direction :" << endl;
	if (allRoom[0] != NULL) {
		cout << "a) go left;  ";
	}
	if (allRoom[1] != NULL){
		cout << "w) go up;  ";
	}
	if (allRoom[2] != NULL) {
		cout << "d) go right;  ";
	}
	if (allRoom[3] != NULL) {
		cout << "s) go down;  ";
	}
	cout<<"n) not going";
	cout << endl;
	while (cin >> input) {
        if(input == "n"){
            break;
        }
		for (int i = 0;i < 4;i++) {
			if (input == ways[i]) {
				if (allRoom[i] != NULL) {
					player.setPreviousRoom(player.getCurrentRoom());
					player.setCurrentRoom(allRoom[i]);
					finish = true;
					break;
				}
			}
		}
		if (finish) {
			break;
		}
	}
	if(input!="n"){
        this->came = false;
        cout<<endl<<endl;
        cout<<"you are entering a room..."<<endl;
        usleep(500000);
	}

}

void Dungeon::startGame() {
	createMap();
	createPlayer();

}


bool Dungeon::checkGameLogic() {
	if (player.getCurrentRoom()->getIsExit() && player.getCurrentRoom()->getObjects()[0]->getCurrentHealth()<=0)
		return false;
	if (player.getCurrentHealth() <= 0)
		return false;
	return true;
}

void Dungeon::chooseAction(vector<Object*> objects) {
    cout<<endl;
    cout<<"==============================="<<endl;
    cout<<endl;
	cout<<"1 : Move"<<endl;
	cout<<"2 : Check Status"<<endl;
	cout<<"3 : Open BackPack"<<endl;
	bool isNPC = false;
	if(objects.size() != 0) {
        if(objects[0]->getTag() == "npc"){
            cout<<"4 : Buy something from the NPC"<<endl;
            isNPC = true;
        }
	}
	int trigger = 0;
	string input;
	cout<<endl;
	while(cin>>input){
        if(input=="1"){                                         // move
            handleMovement();
            break;
        }
        else if(input=="2"){                                    // show status
            player.triggerEvent(&player);
            break;
        }
        else if(input=="3"){                                    // open backpack
            vector<Item> inside = player.getInventory();
            vector<Item> temp ={};
            cout<<endl<<"(1) money * "<<inside[0].getDollar()<<endl;
            temp.push_back(inside[0]);
            inside.erase(inside.begin());
            while (inside.size()>0){
                int countAmount = 1;
                for(int i = 1;i<(int)inside.size();i++){
                    if(inside[0].getName() == inside[i].getName()){
                        countAmount++;
						inside.erase(inside.begin() + i);
						i--;
                    }
                }
                cout<<"("<<temp.size()+1<<") "<<inside[0].getName()<<" * "<<countAmount<<endl;
                temp.push_back(inside[0]);
				inside.erase(inside.begin());
            }
            string useOrNot="";                                 // use item in packpack
            cout<<endl<<"you want to use any of them? (y/n)"<<endl;
            while(cin>>useOrNot){
                if(useOrNot=="y"){
                    string whichToUse="";
                    int use;
                    cout<<"\nwhich one would you like to use?"<<endl;
                    while(cin>>whichToUse){
                        if(whichToUse.length()==1 && (int)whichToUse[0]-(int)'0'<=(int)temp.size() && (int)whichToUse[0]!='0')
                            break;
                    }
                    usleep(500000);
                    use = (int)whichToUse[0]-(int)'0';
                    use--;
                    if(temp[use].getName()=="money" || temp[use].getName()=="PURIFYING LASERGUN" || temp[use].getName()=="revive armor"){
                        cout<<"\nyou can not use this in you backpack!"<<endl;
                    }
                    else if(temp[use].getName()=="unknown healthy potion" || temp[use].getName()=="body strengthening" ||temp[use].getName()=="Devil's egg" ||
                            temp[use].getName()=="small sword"){
                        player.increaseStates(temp[use].getHealth(),temp[use].getAttack(),temp[use].getDefense());
                        player.setCurrentHealth(player.getCurrentHealth()+temp[use].getHealth());
                        if(temp[use].getName()=="small sword"){
                            cout<<"\nsmall sword is armed"<<endl;
                        }
                        else if(temp[use].getName()=="unknown healthy potion"){
                            cout<<"\nyou drank the potion..."<<endl;
                        }
                        else if(temp[use].getName()=="body strengthening"){
                            cout<<"\nyou look a little stronger!!! XD"<<endl;
                        }
                        else if(temp[use].getName()=="Devil's egg"){
                            cout<<"\nyou swallowed the Devil's egg...\nyou sure you won't get a desease?"<<endl;
                        }
                        player.removeItem(temp[use]);
                    }
                    else if(temp[use].getName()=="one-time recover potion"){
                        player.setCurrentHealth(player.getMaxHealth());
                        cout<<"\nall the wounds are healed..."<<endl;
                        player.removeItem(temp[use]);
                    }
                    else if(temp[use].getName()=="time-sensitive prediction"){
                        cout<<"\nwhisper...\nwhisper...\nwhisper...\nGod told me that...\nsomething bad is at the room above...\nbe prepared..."<<endl;
                        player.removeItem(temp[use]);
                    }
                    else if(temp[use].getName()=="devil fruit - dark" || temp[use].getName()=="devil fruit - vibrate"){
                        if(player.getDevilFruit()==false){
                            player.increaseStates(temp[use].getHealth(),temp[use].getAttack(),temp[use].getDefense());
                            player.setCurrentHealth(player.getCurrentHealth()+temp[use].getHealth());
                            cout<<"\noops, you ate the devil fruit! OuO"<<endl;
                            player.removeItem(temp[use]);
                            player.setDevilFruit(true);
                        }
                        else{
                            cout<<"\nYou can ONNNNNNNLLLLLLY eat 1 Devil fruit at a time!!"<<endl;
                        }
                    }
                    else if(temp[use].getName()=="unknown magic paper"){
                        if(player.getDevilFruit()==false){
                            cout<<"you should eat one first!"<<endl;
                        }
                        else {
                            cout<<"\na weird voice appeared..."<<endl;
                            cout<<"Black beard : let me give you some help with the devil fruits!"<<endl;
                            cout<<"you now know how to have 2 devil fruits!"<<endl;
                            player.setDevilFruit(false);
                            player.removeItem(temp[use]);
                        }
                    }
                    player.triggerEvent(&player);
                    break;
                }
                if(useOrNot=="n"){
                    break;
                }
            }
            break;
        }
        else if(input=="4" && isNPC){                           // interact with NPC
            trigger = objects[0]->triggerEvent(&player);
			if (trigger < 10) {
				Item newItem(objects[0]->getCommodityName(trigger), objects[0]->getCommodityHealth(trigger), objects[0]->getCommodityAttack(trigger), objects[0]->getCommodityDefense(trigger), 0);
				player.addItem(newItem);
				cout<<endl<<objects[0]->getName()<<" : sure, here it is."<<endl;
				//Item money = Item("money",0,0,0,objects[0]->getCommodityDollar(trigger));
				//player.removeItem(money);
			}
			break;
        }
        else{
            cout<<"invalid input!!"<<endl;
        }
	}
}


void Dungeon::loading(){
    for(int i=0;i<3;i++){
        cout<<"."<<endl;
        usleep(500000);
    }
    cout<<endl;
}

void Dungeon::handleEvent(vector<Object*> &objects) {
	int choice;
	bool exitRoom = false;
	for (int i = 0;i < (int)objects.size();i++) {
        ///////////////////////////////////////
        //cout<<objects[i]->getAttack()<<endl;
        if(exitRoom){
            break;
        }
        if(player.getCurrentHealth()<=0) break;
        if(objects[i]->getTag()=="npc" && this->came==false){
            cout<<"\n\nA weird guy comes and talks to you"<<endl;
            loading();
            cout<<objects[i]->getScript()<<endl;
            if(objects[i]->getName()=="god"){
                objects[i]->setScript("god : It seems that we've met before...\ngod : You should get going...");
            }
            this->came = true;
            continue;
        }
        if(objects[i]->getTag()=="npc"){
            continue;
        }
        usleep(500000);
		choice = player.getCurrentRoom()->getObjects()[i]->triggerEvent(objects[i]);
		if (objects[i]->getTag() == "item") {
			Item newItem(objects[i]->getName(), objects[i]->getHealth(), objects[i]->getAttack(), objects[i]->getDefense(), objects[i]->getDollar());
            if(objects[i]->getName()!="chest"){
                player.addItem(newItem);
            }
			player.getCurrentRoom()->popObject(objects[i]);
			i--;

		}

		// battle part //
		else if (objects[i]->getTag() == "monster" ){
			if (choice == 0) {
				player.setCurrentRoom(player.getPreviousRoom());
				cout<<"you successfully retreated..."<<endl;
				this->came = true; //
				return;
			}
			else {
                vector <Item> tempInven = player.getInventory();
                for(int j=0;j<(int)tempInven.size();j++){
                    if(tempInven[j].getName()=="PURIFYING LASERGUN"){
                        cout<<"do you want to activate the PURIFYING LASERGUN? (y/n)"<<endl;
                        string x="";
                        while(cin>>x){
                            if(x=="y" || x == "n")
                                break;
                        }
                        if(x=="y"){
                            cout<<"\nLASERRRRRRRRRRRRRRRR!"<<endl;
                            cout<<"Devil eliminated!!!\n"<<endl;
                            objects[i]->setCurrentHealth(0);
                        }
                    }
                }
                if(player.getDevilFruit()){
                    cout << objects[i]->getName()<<": you are... Devilll!!!!\n";
                    cout << "the devil is debuffed" << endl;
                    int* debuff = objects[i]->getDebuff();
                    objects[i]->setCurrentHealth(objects[i]->getCurrentHealth()-debuff[1]);
                    objects[i]->setAttack(objects[i]->getAttack()-debuff[1]);
                    objects[i]->setDefense(objects[i]->getDefense()-debuff[2]);
                }
				while (player.getCurrentHealth() > 0 && objects[i]->getCurrentHealth() > 0) {
                    int temp;
                    temp = objects[i]->getDefense()-player.getAttack();
                    if(temp>=0) temp=-1;
					objects[i]->setCurrentHealth(objects[i]->getCurrentHealth() + temp);
					if (objects[i]->getCurrentHealth() <= 0) {
						cout << "\nyou won!\n" << endl;
						player.getCurrentRoom()->popObject(objects[i]);
						i--;
						break;
					}
					temp = player.getDefense()-objects[i]->getAttack();
					if(temp>=0) temp=-1;
					player.setCurrentHealth(player.getCurrentHealth() + temp);
				}
				if (player.getCurrentHealth() <= 0) {
					vector <Item> tmp = player.getInventory();
					for (int i = 0;i < (int)tmp.size();i++) {
						if (tmp[i].getName() == "revive armor") {
                            cout<<"\nyou are dead during the fight..."<<endl;
                            cout<<"revive armor activated..."<<endl;
                            cout<<"you are sent to the previous room"<<endl;
							player.setCurrentRoom(player.getPreviousRoom());
							player.setCurrentHealth(player.getMaxHealth());
							exitRoom = true;
							player.removeItem(player.getInventory()[i]);
							break;
						}
					}
				}
			}
		}


	}
}


void Dungeon::runDungeon() {
	startGame();
	cout<<"game started!\nyou entered a room..."<<endl;
	while (checkGameLogic()) {
		handleEvent(player.getCurrentRoom()->getObjects());
		if(player.getCurrentHealth()<=0 || rooms[10].getObjects()[0]->getCurrentHealth()<=0)
            break;
		chooseAction(player.getCurrentRoom()->getObjects());
	}
	if(player.getCurrentHealth()>0){
        cout<<"Congratulations!\nYou are such a strong fighter\nYOU WON!"<<endl;
	}
	else{
        cout<<"oops, you are dead\ntry again next time!"<<endl;
	}
}

