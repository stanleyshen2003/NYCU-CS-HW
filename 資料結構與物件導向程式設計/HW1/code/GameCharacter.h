#ifndef GAMECHARACTER_H_INCLUDED
#define GAMECHARACTER_H_INCLUDED

#include <iostream>
#include <string>
#include "Object.h"
using namespace std;

class GameCharacter: public Object
{
private:
    int maxHealth;
    int currentHealth;
    int attack;
    int defense;

public:
    GameCharacter();
    GameCharacter(string,string,int,int,int);
    bool checkIsDead();
    int takeDamage(int);

    /* Set & Get function*/
    void setMaxHealth(int);
    void setCurrentHealth(int) override;
    void setAttack(int);
    void setDefense(int);
    int getMaxHealth() const;
    int getCurrentHealth() const override;
    int getAttack() const;
    int getDefense() const;
};
#endif // GAMECHARACTER_H_INCLUDED
