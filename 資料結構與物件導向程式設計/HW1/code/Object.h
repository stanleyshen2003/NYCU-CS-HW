#ifndef OBJECT_H_INCLUDED
#define OBJECT_H_INCLUDED

#include <iostream>
#include <string>
#include <vector>
using namespace std;

class Object
{
private:
    string name;
    string tag;
public:
    Object();
    Object(string,string);


    virtual int triggerEvent(Object*){cout<<"call base";return 1;}
    virtual int getHealth() const{return 1;}
    virtual int getAttack() const{return 1;}
    virtual int getDefense() const{return 1;}
    virtual int getDollar() const{return 1;}
    virtual void setMoney(int){}
    virtual int getMoney(){return 1;}
    virtual string getCommodityName(int){return "";}
    virtual int getCommodityHealth(int){return 1;}
    virtual int getCommodityAttack(int){return 1;}
    virtual int getCommodityDefense(int){return 1;}
    virtual int getCommodityDollar(int){return 1;}
    virtual int getCurrentHealth() const{return 1;}
    virtual void setAttack(int){}
    virtual void setDefense(int){}
    virtual void setCurrentHealth(int) {}
    virtual string getScript() const{return "";}
    virtual void setScript(string){}
    virtual int* getDebuff(){}
    /////////////////////////////////////
    /* Set & Get function*/
    void setName(string);
    void setTag(string);
    string getName() const;
    string getTag() const;
};

#endif // OBJECT_H_INCLUDED
