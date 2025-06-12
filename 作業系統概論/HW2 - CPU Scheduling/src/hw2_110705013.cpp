#include <iostream>
using namespace std;

int N,M;

struct process{
    int arr;
    int bst;
    int fnh;
    int index;
    int realbst;
    int inQtime;
    process* next;
};

struct queue{
    process* process_ll_first;
    process* process_ll_last;
    int num;
    int mode;
    int timeQuantum;
    int nowid;
    int Qleft;
};

process* newnode(int burst, int arrive, int index){
    process* temp = new process;
    temp -> next = NULL;
    temp -> bst = burst;
    temp -> arr = arrive;
    temp -> index = index;
    temp -> realbst = burst;
    temp -> inQtime = arrive;
    return temp;
}

process* newnode(int burst, int arrive, int index, int realbst, int timestamp){
    process* temp = new process;
    temp -> next = NULL;
    temp -> bst = burst;
    temp -> arr = arrive;
    temp -> index = index;
    temp -> realbst = realbst;
    temp -> inQtime = timestamp;
    return temp;
}

void insertQ(queue *Q, int burst, int arrive, int index){
    if(Q->process_ll_last == NULL)
        Q->process_ll_first = Q->process_ll_last = newnode(burst,arrive,index);
    else{
        process* temp = newnode(burst,arrive,index);
        Q -> process_ll_last -> next = temp;
        Q -> process_ll_last = temp;
    }
    Q -> num += 1;
}

void insertQ(queue *Q, int burst, int arrive, int index, int realbst, int timestamp){
    if(Q->process_ll_last == NULL)
        Q->process_ll_first = Q->process_ll_last = newnode(burst,arrive,index, realbst, timestamp);
    else{
        process* temp = newnode(burst,arrive,index, realbst, timestamp);
        Q -> process_ll_last -> next = temp;
        Q -> process_ll_last = temp;
    }
    Q -> num += 1;
}

void pop_first(queue& Q){
    if(Q.num==0){
        cout<<"pop error, no element\n";
        return;
    }
    else if (Q.num==1){
        Q.process_ll_first = Q.process_ll_last = NULL;
    }
    else{
        process* temp = Q.process_ll_first;
        Q.process_ll_first = Q.process_ll_first->next;
        delete(temp);
    }
    Q.num--;
}

void pop_idx(queue& Q, int index){
    if(Q.num==0){
        cout<<"pop error, no element\n";
        return;
    }
    else if (Q.num==1){
        delete(Q.process_ll_first);
        Q.process_ll_first = Q.process_ll_last = NULL;
    }
    else{
        if(Q.process_ll_first->index==index){
            process* temp = Q.process_ll_first;
            Q.process_ll_first = Q.process_ll_first->next;
            delete(temp);
        }
        else{
            process* temp = Q.process_ll_first;
            for(process* temp2 = temp->next;temp2!=NULL;temp2 = temp2->next){
                if(temp2->index == index){
                    process* temp3 = temp2;

                    temp->next = temp2->next;
                    if(temp->next == NULL){
                        Q.process_ll_last = temp;
                    }
                    delete(temp3);
                    break;
                }
                temp = temp2;
            }
        }
    }
    Q.num--;
    
}

process* getrunning(queue & q, int index){
    for(process* i = q.process_ll_first; i != NULL; i = i->next){
        if(i->index == index)
            return i;
    }
    return NULL;
}

void select_run(queue* Qs, int &timestamp, queue& finish){
    int run = -1;
    int Qindex = 0;
    timestamp ++;
    for(Qindex = 0; Qindex < N; Qindex++){
        if(Qs[Qindex].num!=0){
            break;
        }
    }
    if(Qindex == N){
        return;
    } 
    for(int i=0; i<N; i++){
        if(i!=Qindex){
            if(Qs[i].nowid != -1){
                if(Qs[i].mode == 2){
                    Qs[i].Qleft = Qs[i].timeQuantum;
                }
                process* recent = getrunning(Qs[i], Qs[i].nowid);
                if(i<N-1){
                    insertQ(&Qs[i+1], recent->bst, recent->arr, recent->index, recent->realbst, timestamp-1);
                    
                }
                else{
                    insertQ(&Qs[i], recent->bst, recent->arr, recent->index, recent->realbst, timestamp-1);
                }
                pop_idx(Qs[i], Qs[i].nowid);
                Qs[i].nowid = -1;
            }
        }
    }
    if(Qs[Qindex].mode == 0){               // mode 0 FIFO
        Qs[Qindex].nowid = Qs[Qindex].process_ll_first->index;
        //cout<<"run"<<Qindex<<": "<<Qs[Qindex].process_ll_first->index<<endl;
        if(--Qs[Qindex].process_ll_first->bst == 0){
            insertQ(&finish, Qs[Qindex].process_ll_first->realbst, Qs[Qindex].process_ll_first->arr, Qs[Qindex].process_ll_first->index);
            finish.process_ll_last->fnh = timestamp;
            pop_first(Qs[Qindex]);
            M--;
            Qs[Qindex].nowid = -1;
        } 
        
    }
    else if(Qs[Qindex].mode == 1){          // mode 1 Shortest Remaining Time First
        process* best = Qs[Qindex].process_ll_first;
        for(process* temp = Qs[Qindex].process_ll_first;temp!=NULL;temp = temp->next){
            if(best->bst > temp->bst){
                best = temp;
            }
            else if(best->bst == temp->bst){
                if(best->index > temp->index){
                    best = temp;
                }
            }
        }
        if(Qs[Qindex].nowid != best->index && Qs[Qindex].nowid!=-1 && Qindex < N-1){
            process* recent = getrunning(Qs[Qindex], Qs[Qindex].nowid);
            insertQ(&Qs[Qindex+1], recent->bst, recent->arr, recent->index, recent->realbst, timestamp-1);
            pop_idx(Qs[Qindex], Qs[Qindex].nowid);
        }
        Qs[Qindex].nowid = best->index;
        //cout<<"run"<<Qindex<<": "<<best->index<<endl;
        best->bst -= 1;
        if(best->bst==0){
            insertQ(&finish, best->realbst, best->arr, best->index);
            finish.process_ll_last->fnh = timestamp;
            pop_idx(Qs[Qindex], best->index);
            M--;
            Qs[Qindex].nowid = -1;
        }
        
    }
    else{
        process* selected = Qs[Qindex].process_ll_first;
        for(process* temp = selected->next;temp!=NULL;temp = temp->next){
            if(selected->inQtime > temp->inQtime){
                selected = temp;
            }
            else if (selected->inQtime == temp->inQtime && temp->arr > selected->arr){
                selected = temp;
            }
        }
        Qs[Qindex].nowid = selected->index;
        //cout<<"run"<<Qindex<<": "<<selected->index<<endl;
        if(Qs[Qindex].Qleft == 1 && selected->bst>1){
            selected->bst --;
            process* bye = selected;
            if(Qindex < N - 1){
                insertQ(&Qs[Qindex+1], bye->bst, bye->arr, bye->index, bye->realbst, timestamp);
            }
            else{
                insertQ(&Qs[Qindex], bye->bst, bye->arr, bye->index, bye->realbst, timestamp);
            }
            pop_idx(Qs[Qindex], selected->index);
            Qs[Qindex].Qleft = Qs[Qindex].timeQuantum;
            Qs[Qindex].nowid = -1;
        }
        else if(--selected->bst == 0){
            insertQ(&finish, selected->realbst, selected->arr, selected->index);
            finish.process_ll_last->fnh = timestamp;
            pop_idx(Qs[Qindex], selected->index);
            M--;
            Qs[Qindex].Qleft = Qs[Qindex].timeQuantum;
            Qs[Qindex].nowid = -1;
        }
        else{
            Qs[Qindex].Qleft --;
        }
    }     
}

void insert_arrive(queue& first, queue& notarrived, int timestamp){
    process* temp = notarrived.process_ll_first;
    while(temp != NULL){
        if(temp->arr <= timestamp){
            insertQ(&first,temp->bst,temp->arr, temp->index);
            temp = temp->next;
            pop_first(notarrived);
        }
        else break;
    }
}

int main(){

    // intput N and M
    cin>>N>>M;
    int total = M;
    queue* Qs;
    queue finish;
    Qs = new queue[N];
    
    int mode, timeQuantum;
    int timestamp = 0;
    for(int i = 0; i < N; i++){
        cin >> mode >> timeQuantum;
        Qs[i].mode = mode;
        Qs[i].timeQuantum = timeQuantum;
        Qs[i].num = 0;
        Qs[i].nowid = -1;
        Qs[i].Qleft = timeQuantum;
    }
    // store process
    int arrive,burst;
    queue notarrived;
    notarrived.mode = 0;
    for(int i = 0; i < M; i++){
        cin >> arrive >> burst;
        insertQ(&notarrived, burst, arrive, i);
    }
    
    while(M){

        insert_arrive(Qs[0], notarrived, timestamp);
        // cout<< "timestamp: "<<timestamp<<endl;
        // cout<< "M: " << M<<endl;
        // cout<<"finish: "<<finish.num<<endl;
        // cout<<"# in Q: "<<Qs[0].num<<endl;
        // cout<<"Qs after insert"<<endl;
        // for(int i=0;i<N;i++){
        //     cout<<"Q"<<i<<": ";
        //     for(process* temp = Qs[i].process_ll_first;temp!=NULL;temp = temp->next){
        //         cout<< temp->index <<" ";
        //     }
        //     cout << endl;
        // }
        
        //cout<<endl;
        
        select_run(Qs, timestamp, finish);
        // cout<<"run done"<<endl;
        // cout << "not arrived: " << notarrived.num<<endl;
        // cout<<"-----------------------------"<<endl;
        
    }

    int total_wait_time = 0;
    int total_turn_around_time = 0;
    
    /////// finish ////// print ////////
    // for(process* temp = finish.process_ll_first;temp!=NULL;temp = temp->next){
    //     cout<< temp->bst <<" "<< temp->index<<" "<<temp->fnh<<endl;
    // }
    // cout<<endl;
    
    ////// end //////////
    for(int i=0;i<total;i++){
        for(process* temp = finish.process_ll_first;temp!=NULL;temp = temp->next){
            if(temp->index == i){
                cout << temp->fnh - temp->arr - temp->bst << " " << temp->fnh - temp->arr << endl;
                total_wait_time += temp->fnh - temp->arr - temp->bst;
                total_turn_around_time += temp->fnh - temp->arr;
                break;
            }
        }
    }
    cout << total_wait_time << endl;
    cout << total_turn_around_time << endl;
    delete(Qs);
    return 0;

}