#ifndef ITEM_H_INCLUDED
#define ITEM_H_INCLUDED

#include <iostream>
#include <string>
#include <vector>
#include "Object.h"
#include "Player.h"
using namespace std;

class Player;

class Item: public Object
{
private:
    int health,attack,defense,dollar;
public:
    Item();
    Item(string, int, int, int, int);

    /* Virtual function that you need to complete    */
    /* In Item, this function should deal with the   */
    /* pick up action. You should add status to the  */
    /* player.                                       */
    int triggerEvent(Object*) override;
    /* Set & Get function*/
    int getHealth() const;
    int getAttack() const;
    int getDefense() const;
    int getDollar() const;
    void setHealth(int);
    void setAttack(int);
    void setDefense(int);
    void setDollar(int);
};

#endif // ITEM_H_INCLUDED
