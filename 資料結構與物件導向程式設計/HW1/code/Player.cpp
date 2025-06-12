#include "Player.h"

Player::Player() :currentRoom(NULL), previousRoom(NULL), inventory(0) {
}
Player::Player(string name, int health, int attack, int defense) : GameCharacter(name, "Player", health, attack, defense), currentRoom(NULL), previousRoom(NULL),inventory(0),devil_fruit(false) {}

void Player::addItem(Item additem) {
	if (additem.getName() == "money" && inventory.size()!=0) {
		inventory[0].setDollar(inventory[0].getDollar() + additem.getDollar());
	}
	else {
		inventory.push_back(additem);
	}
}

void Player::removeItem(Item &item){
    if(item.getName()=="money"){
        inventory[0].setDollar(inventory[0].getDollar()-item.getDollar());
        return;
    }
    for(int i=1;i<inventory.size();i++){
        if(inventory[i].getName()==item.getName()){
            inventory.erase(inventory.begin()+i);
            break;
        }
    }
}

void Player::increaseStates(int health, int attack, int defense) {
	setMaxHealth(getMaxHealth() + health);
	setAttack(getAttack() + attack);
	setDefense(getDefense() + defense);
}
void Player::changeRoom(Room* next) {
	previousRoom = currentRoom;
	currentRoom = next;
}

/* Virtual function that you need to complete   */
/* In Player, this function should show the     */
/* status of player.                            */
int Player::triggerEvent(Object* obj) {
	cout << endl<<"This is your status" << endl;
	cout << "Health : " << getCurrentHealth() << "/" << getMaxHealth() << endl;
	cout << "ATK    : " << getAttack() << endl;
	cout << "DEF    : " << getDefense() << endl;
	return 1;
}

/* Set & Get function*/
void Player::setDevilFruit(bool fruit) {
	devil_fruit = fruit;
}
void Player::setCurrentRoom(Room* currentRoom) {
	this->currentRoom = currentRoom;
}

void Player::setPreviousRoom(Room* previousRoom) {
	this->previousRoom = previousRoom;
}

void Player::setInventory(vector<Item> inventory) {
	this->inventory = inventory;
}

Room* Player::getCurrentRoom() const{
	return currentRoom;
}
Room* Player::getPreviousRoom() const {
	return previousRoom;
}
vector<Item> Player::getInventory() const {
	return inventory;
}
bool Player::getDevilFruit() const {
	return devil_fruit;
}
void Player::setMoney(int money) {
	addItem(Item("money", 0, 0, 0, money - inventory[0].getDollar()));
}

int Player::getMoney() {
	return inventory[0].getDollar();
}
