
#include <iostream>
#include "Stream_Handler.h"

int main(int argc, char **argv)
{
	if (argc != 2){
		cout<<"test as: ./Test_Stream_Handler filename"<<endl;
		return 1;
	} 	

	Stream_Handler stream(argv[1]);
	if (stream.is_open()){
		cout <<"object created using constructor with filename"<<endl;
		cout <<"file was open succesfully as tested with the function is_open"<<endl;
		
	} else {
		cout <<"object created but problem opening the filename detected thorugh function is_open"<<endl;
	}
	
	cout<<"testing get_next()"<<endl;
	cout<<"trying to obtain 3 elements one by one"<<endl;
	
	svm_problem* next_problem;
	svm_node* x;
	double y;
	int T = 3;
	for(int i=0;i<T;i++){
	next_problem = stream.get_next();
	cout<<"read:"<<next_problem->l<<" new example(s)"<<endl;
	x=next_problem->x[0];
	y=next_problem->y[0];
	cout<<"x="<<endl;
	while(x->index > 0){
		cout<<" "<<x->index<<" : "<<x->value;
		x++;
	}
	cout<<" y="<<y<<endl;
	}
	
	cout<<"testing get_next(int nex)"<<endl;
	cout<<"trying to obtain 3 new elements in just one call"<<endl;
	
	T = 3;
	next_problem = stream.get_next(T);
	cout<<"read:"<<next_problem->l<<" new example(s)"<<endl;

	
	for(int i=0;i<next_problem->l;i++){
		x=next_problem->x[i];
		y=next_problem->y[i];
		cout<<"x=";
		while(x->index > 0){
			cout<<" "<<x->index<<" : "<<x->value;
			x++;
		}
		cout<<" y="<<y<<endl;
	}
	cout<<"testing numex_read()"<<endl;
	cout<<"we have read: "<<stream.numex_read()<<" examples"<<endl;
	
	cout<<"testing restart()"<<endl;
	if(stream.restart())
		cout<<"stream re-started"<<endl;
	else
		cout<<"re-started failed"<<endl;
	
	T = 2;
	next_problem = stream.get_next(T);
	cout<<"reading:"<<next_problem->l<<" new example(s)"<<endl;
	
	for(int i=0;i<next_problem->l;i++){
		x=next_problem->x[i];
		y=next_problem->y[i];
		cout<<"x =";
		while(x->index > 0){
			cout<<" "<<x->index<<" : "<<x->value;
			x++;
		}
		cout<<" y = "<<y<<endl;
	}
	
	cout<<"testing close()"<<endl;
	if(stream.close())
		cout<<"stream closed succesfully"<<endl;
	
	cout<<"opening again:"<<endl;
	stream.open(argv[1]);	
	T = 4;
	next_problem = stream.get_next(T);
	cout<<"reading:"<<next_problem->l<<" new example(s)"<<endl;
	
	for(int i=0;i<next_problem->l;i++){
		x=next_problem->x[i];
		y=next_problem->y[i];
		cout<<"x =";
		while(x->index > 0){
			cout<<" "<<x->index<<" : "<<x->value;
			x++;
		}
		cout<<" y = "<<y<<endl;
	}
	
	cout<<"testing info()"<<endl;
	stream_info* info_stream = stream.info();
	if(info_stream==NULL){
		cout<<"info() failed"<<endl;
	} else {
		print_stream_info(info_stream);
	}
	
	return 0;
	
}