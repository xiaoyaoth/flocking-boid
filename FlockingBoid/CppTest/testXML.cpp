#include <iostream>
#include "lib\rapidxml-1.13\rapidxml.hpp"
#include "lib\rapidxml-1.13\rapidxml_utils.hpp"
#include "lib\rapidxml-1.13\rapidxml_print.hpp"
#include <iostream>
#include <iomanip>

using namespace rapidxml;

int writeXMLExample()
{ 
	xml_document<> doc; 
	xml_node<>* rot = doc.allocate_node(rapidxml::node_pi,doc.allocate_string("xml version='1.0' encoding='utf-8'")); 
	doc.append_node(rot); 
	xml_node<>* node = doc.allocate_node(node_element,"config","information"); 
	xml_node<>* color = doc.allocate_node(node_element,"color",NULL); 
	doc.append_node(node); 
	node->append_node(color); 
	color->append_node(doc.allocate_node(node_element,"red","0.1")); 
	color->append_node(doc.allocate_node(node_element,"green","0.1")); 
	color->append_node(doc.allocate_node(node_element,"blue","0.1")); 
	color->append_node(doc.allocate_node(node_element,"alpha","1.0")); 

	xml_node<>* size = doc.allocate_node(node_element,"size",NULL); 
	size->append_node(doc.allocate_node(node_element,"x","640")); 
	size->append_node(doc.allocate_node(node_element,"y","480")); 
	node->append_node(size); 

	xml_node<>* mode = doc.allocate_node(rapidxml::node_element,"mode","screen mode"); 
	mode->append_attribute(doc.allocate_attribute("fullscreen","false")); 
	node->append_node(mode); 

	std::string text; 
	rapidxml::print(std::back_inserter(text), doc, 0); 

	std::cout<<text<<std::endl; 

	std::ofstream out("config.xml"); 
	out << doc; 

	system("PAUSE"); 
	return EXIT_SUCCESS; 
}
int readXMLExample(){
	file<> fdoc("config.xml");
	std::cout<<fdoc.data()<<std::endl;
	xml_document<> doc;
	doc.parse<0>(fdoc.data());

	std::cout<<doc.name()<<std::endl;

	//! 获取根节点
	xml_node<>* root = doc.first_node();
	std::cout<<root->name()<<std::endl;

	//! 获取根节点第一个节点
	xml_node<>* node1 = root->first_node();
	std::cout<<node1->name()<<std::endl;

	xml_node<>* node11 = node1->first_node();
	std::cout<<node11->name()<<std::endl;
	std::cout<<node11->value()<<std::endl;

	//! 添加之后再次保存
	//需要说明的是rapidxml明显有一个bug
	//那就是append_node(doc.allocate_node(node_element,"h","0"));的时候并不考虑该对象是否存在!
	xml_node<>* size = root->first_node("size");
	size->append_node(doc.allocate_node(node_element,"w","0"));
	size->append_node(doc.allocate_node(node_element,"h","0"));

	std::string text;
	rapidxml::print(std::back_inserter(text),doc,0);

	std::cout<<text<<std::endl;

	std::ofstream out("config.xml");
	out << doc;

	system("PAUSE");
	return EXIT_SUCCESS;
}


int writeXML(){
	xml_document<> doc;
	xml_node<>* rot = doc.allocate_node(rapidxml::node_pi,doc.allocate_string("xml version='1.0' encoding='utf-8'"));
	doc.append_node(rot);

	xml_node<> *root, *node;
	root = doc.allocate_node(node_element, "config", NULL);
	doc.append_node(root);
	node = doc.allocate_node(node_element, "AGENT_NO", "1024");
	root->append_node(node);
	node = doc.allocate_node(node_element, "repetition", "1000");
	root->append_node(node);

	std::ofstream out("config.xml"); 
	out << doc; 

	return EXIT_SUCCESS; 
}
int scanNode(xml_node<> *parant, int dent){
	dent++;
	xml_node<> *child = parant->first_node();
	for(; child!=NULL; child=child->next_sibling()){
		//std::cout<<std::setw(10)<<std::setfill('c');
		for(int i=0; i<dent; i++)
			std::cout<<"\t";
		std::cout<<child->name()<<" ";
		std::cout<<child->value()<<std::endl;
		scanNode(child, dent);
	}
	return 0;
}
int readXML(){
	file<> fdoc("config.xml");
	std::cout<<fdoc.data()<<std::endl;
	xml_document<> doc;
	doc.parse<0>(fdoc.data());
	xml_node<>* root = doc.first_node();
	std::cout<<root->name()<<std::endl;
	scanNode(root, 0);
	system("PAUSE");
	return EXIT_SUCCESS;
}

int mainXML() {
	writeXML();
	readXML();
	return EXIT_SUCCESS;
}
