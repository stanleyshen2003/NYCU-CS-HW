#ifndef PLAYER_H_INCLUDED
#define PLAYER_H_INCLUDED

#include <iostream>
#include <string>
#include <vector>
#include "GameCharacter.h"
#include "Room.h"
#include "Item.h"

using namespace std;

class Item;

class Player: public GameCharacter
{
private:
    bool devil_fruit = false;
    Room* currentRoom;
    Room* previousRoom;
    vector<Item> inventory;
public:
    Player();
    Player(string, int, int, int);
    void addItem(Item);
    void increaseStates(int,int,int);
    void changeRoom(Room*);
    void setDevilFruit(bool);
    bool getDevilFruit() const;
    /* Virtual function that you need to complete   */
    /* In Player, this function should show the     */
    /* status of player.                            */
    int triggerEvent(Object*) override;
    void setMoney(int);
    int getMoney();
    void removeItem(Item&);
    /* Set & Get function*/
    void setCurrentRoom(Room*);
    void setPreviousRoom(Room*);
    void setInventory(vector<Item>);
    Room* getCurrentRoom() const;
    Room* getPreviousRoom() const;
    vector<Item> getInventory() const;
};

#endif // PLAYER_H_INCLUDED
