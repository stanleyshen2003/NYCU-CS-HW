#include "Object.h"

Object::Object():name("no_name"),tag("no_tag"){}
Object::Object(string name,string tag):name(name),tag(tag){}


void Object::setName(string newName) {
	name = newName;
}

void Object::setTag(string newTag) {
	tag = newTag;
}
string Object::getName() const {
	return name;
}
string Object::getTag() const {
	return tag;
}

