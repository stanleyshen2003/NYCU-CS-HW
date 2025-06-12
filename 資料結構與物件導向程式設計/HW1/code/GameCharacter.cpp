#include "GameCharacter.h"

GameCharacter::GameCharacter():Object(),currentHealth(100),attack(10),defense(10){}

GameCharacter::GameCharacter(string name, string tag, int currentHealth, int attack, int defense):Object(name,tag),
								currentHealth(currentHealth),maxHealth(currentHealth), attack(attack), defense(defense) {}

bool GameCharacter::checkIsDead() {
	if (currentHealth == 0)
		return true;
	return false;
}
int GameCharacter::takeDamage(int damage) {
	currentHealth -= damage;
	return currentHealth;
}
void GameCharacter::setMaxHealth(int newMax) {
	maxHealth = newMax;
}
void GameCharacter::setCurrentHealth(int newCurrent) {
	currentHealth = newCurrent;
}
void GameCharacter::setAttack(int newAtk) {
	attack = newAtk;
}
void GameCharacter::setDefense(int newDef) {
	defense = newDef;
}

int GameCharacter::getMaxHealth() const{
	return maxHealth;
}
int GameCharacter::getCurrentHealth() const{
	return currentHealth;
}
int GameCharacter::getAttack() const {
	return attack;
}
int GameCharacter::getDefense() const {
	return defense;
}
