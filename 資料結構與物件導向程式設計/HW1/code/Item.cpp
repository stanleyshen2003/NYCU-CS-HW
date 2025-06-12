#include "Item.h"

Item::Item(){
}

Item::Item(string name, int health, int attack, int defense, int dollar):Object(name,"item"),health(health),attack(attack),defense(defense),dollar(dollar) {}


/* Virtual function that you need to complete    */
/* In Item, this function should deal with the   */
/* pick up action. You should add status to the  */
/* player.                                       */
int Item::triggerEvent(Object* obj){
    if(getName()=="chest")
        cout<<"You found and opened a chest!"<<endl;
	else if (getName() == "money") {
		cout << "\nyou received " << getDollar() << " dollars." << endl;
	}
    else{
        cout<<"\nYou picked up "<<getName()<<endl;
    }
	return 1;
}

/* Set & Get function*/
int Item::getHealth() const{
	return health;
}
int Item::getAttack() const {
	return attack;
}
int Item::getDefense() const {
	return defense;
}
int Item::getDollar() const {
	return dollar;
}
void Item::setHealth(int health) {
	this->health = health;
}
void Item::setAttack(int attack) {
	this->attack = attack;
}
void Item::setDefense(int defense) {
	this->defense = defense;
}
void Item::setDollar(int dollar) {
	this->dollar = dollar;
}
