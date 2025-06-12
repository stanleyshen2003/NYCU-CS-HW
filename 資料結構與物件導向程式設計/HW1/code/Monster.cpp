#include "Monster.h"
#include "Player.h"
#include <string>
Monster::Monster() {

}

Monster::Monster(string name,string occupation,string script, int dehealth, int deattack, int dedefense):GameCharacter(name,"monster",50,12,12){
    this->occupation = occupation;
    this->debuff = new int[3];
    this->debuff[0] = dehealth;
    this->debuff[1] = deattack;
    this->debuff[2] = dedefense;
    this->script = script;

    if(occupation == "smallDevil"){
        this->setMaxHealth(200);
        this->setCurrentHealth(200);
        this->setAttack(40);
        this->setDefense(40);
    }
    else if(occupation== "guardian"){
        this->setMaxHealth(700);
        this->setCurrentHealth(700);
        this->setAttack(140);
        this->setDefense(140);
    }
    else if(occupation=="boss"){
        this->setMaxHealth(999999);
        this->setCurrentHealth(999999);
        this->setAttack(999999);
        this->setDefense(999999);
    }
}

int* Monster::getDebuff(){
    return this->debuff;
}
string Monster::getScript(){
    return this->script;
}

/* Virtual function that you need to complete   */
/* In Monster, this function should deal with   */
/* the combat system.                           */
int Monster::triggerEvent(Object* obj) {
	cout << "you bumped into " << getName() << "." << endl;
	cout << script;
	cout << "do you want to have a fight? (y/n)" << endl;
	string fight;
	cin >> fight;
	while (fight != "y" && fight != "n") {
		cout << "invalid input!" << endl;
		cout << "do you want to have a fight? (y/n)" << endl;
		cin >> fight;
	}
	if (fight == "y")
		return 1;
	else {
		return 0;
	}


}
