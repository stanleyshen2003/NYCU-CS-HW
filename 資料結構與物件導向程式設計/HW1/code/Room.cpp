#include "Room.h"

Room::Room():upRoom(NULL),downRoom(NULL),leftRoom(NULL),rightRoom(NULL),isExit(0),index(0) {
	this->objects = {};
}

Room::Room(bool isExit, int index, vector<Object*> objects): upRoom(NULL), downRoom(NULL), leftRoom(NULL), rightRoom(NULL),isExit(isExit),index(index),objects(objects){}

/*pop out the specific object, used when the interaction is done*/
bool Room::popObject(Object* object) {
	for (int i = 0;i < objects.size();i++) {
		if (objects[i] == object) {
			objects.erase(objects.begin()+i);
			return true;
		}
	}
	return false;
}

/* Set & Get function*/
void Room::setUpRoom(Room* upRoom) {
	this->upRoom = upRoom;
}
void Room::setDownRoom(Room* downRoom) {
	this->downRoom = downRoom;
}
void Room::setLeftRoom(Room* leftRoom) {
	this->leftRoom = leftRoom;
}
void Room::setRightRoom(Room* rightRoom) {
	this->rightRoom = rightRoom;
}

void Room::setAllRoom(Room* leftRoom, Room* upRoom, Room* rightRoom, Room* downRoom) {
	this->upRoom = upRoom;
	this->downRoom = downRoom;
	this->leftRoom = leftRoom;
	this->rightRoom = rightRoom;
}

void Room::setIsExit(bool isExit) {
	this->isExit = isExit;
}
void Room::setIndex(int index) {
	this->index = index;
}
void Room::setObjects(vector<Object*> &objects) {
    this->objects = vector<Object*>(objects.size());
	for(int i=0;i<objects.size();i++){
        this->objects[i] = objects[i];
	}
	//cout<<&this->objects[1]<<" copy"<<endl;
	//cout<<&objects[1]<<endl;
}
bool Room::getIsExit() const{
	return isExit;
}
int Room::getIndex() const {
	return index;
}
vector<Object*>& Room::getObjects() {
	return objects;
}
Room* Room::getUpRoom() const {
	return upRoom;
}
Room* Room::getDownRoom() const {
	return downRoom;
}
Room* Room::getLeftRoom() const {
	return leftRoom;
}
Room* Room::getRightRoom() const {
	return rightRoom;
}


