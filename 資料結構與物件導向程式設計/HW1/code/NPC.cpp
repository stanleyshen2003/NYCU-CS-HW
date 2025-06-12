#include "NPC.h"
NPC::NPC() :script(""), commodity(0), GameCharacter("no_name", "NPC", 0, 0, 0) {
}

NPC::NPC(string name, string script, vector<Item> commodity) : GameCharacter(name, "npc", 0, 0, 0), script(script), commodity(commodity) {
}

/*print all the Item in this NPC*/
void NPC::listCommodity() {
	cout << "here are all the items I have :"<<endl;
	for (int i = 0;i < (int)commodity.size();i++) {
		cout << i+1 << ") " << commodity[i].getName() << endl;;
	}
	cout << commodity.size()+1 <<") not buying..."<<endl;
}

int NPC::getCommodityDollar(int i){
    return commodity[i].getDollar();
}

/* Virtual function that you need to complete   */
/* In NPC, this function should deal with the   */
/* transaction in easy implementation           */
int NPC::triggerEvent(Object* player) {
    if(commodity.size()==0){
        cout<<endl<<getName()<<" : Um... I have nothing to sell XD"<<endl;
        cout<<getName()<<" : the only thing I could say is..."<<endl;
        cout<<script;
        cout<<endl;
        return 200;
    }
	string buy = "";
	cout<<endl<<getName() << " : do you want to buy anything?" << endl<<endl;
	listCommodity();
	cout <<endl<<getName()<< " : which one do you want?" << endl;
	buy = "";
	cin >> buy;
	while (buy.length() > 1) {
		cin >> buy;
	}
	int choice;
	choice = buy[0] - '1';
	if (choice < commodity.size()) {
		if (commodity[choice].getDollar() > player->getMoney()) {
			cout << "not enough money!!" << endl;
		}
		else {
			player->setMoney(player->getMoney() - commodity[choice].getDollar());
			return choice;
		}
	}
	return 100;
}

string NPC::getCommodityName(int i) {
	return commodity[i].getName();
}
int NPC::getCommodityHealth(int i) {
	return commodity[i].getHealth();
}
int NPC::getCommodityAttack(int i) {
	return commodity[i].getAttack();
}
int NPC::getCommodityDefense(int i) {
	return commodity[i].getDefense();
}

/* Set & Get function*/
void NPC::setScript(string script) {
	this->script = script;
}
void NPC::setCommodity(vector<Item> commodity) {
	this->commodity = commodity;
}
string NPC::getScript() const {
	return script;
}
vector<Item> NPC::getCommodity() const {
	return commodity;
}
