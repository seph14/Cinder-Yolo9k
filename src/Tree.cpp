//
//  Tree.cpp
//  StreetViewAnalyzer
//
//  Created by HAOZHE LI on 08/04/2019.
//

#include "Tree.h"

using namespace ml;
using namespace std;

float ml::get_hierarchy_probability(float *x, tree *hier, int c, int stride) {
    float p = 1;
    while(c >= 0){
        p = p * x[c*stride];
        c = hier->parent[c];
    }
    return p;
}

float sig(float z){
    return 1.f / (1.f + (float)exp(-z));
}

void ml::hierarchy_predictions(double *predictions, int n, tree *hier, int only_leaves, int stride) {
    int j;
    for(j = 0; j < n; ++j){
        int parent = hier->parent[j];
        if(parent >= 0){
            predictions[j*stride] *= predictions[parent*stride];
        }
    }
    if(only_leaves){
        for(j = 0; j < n; ++j){
            if(!hier->leaf[j]) predictions[j*stride] = 0;
        }
    }
}

int ml::hierarchy_top_prediction(double *predictions, tree *hier, float thresh, int stride) {
    float p     = 1;
    int group   = 0;
    int i;
    while(1){
        float max = 0;
        int max_i = 0;
        
        for(i = 0; i < hier->group_size[group]; ++i){
            int index = i + hier->group_offset[group];
            float val = (predictions[(i + hier->group_offset[group])*stride]) ;
            if(val > max){
                max_i = index;
                max   = val;
            }
        }
        
        if(p*max > thresh){
            p     = p*max;
            group = hier->child[max_i];
            if(hier->child[max_i] < 0) return max_i;
        } else if (group == 0){
            return max_i;
        } else {
            return hier->parent[hier->group_offset[group]];
        }
    }
    return 0;
}

tree* ml::read_tree(ci::DataSourceRef data) {
    tree t = {0};
    
    auto stream = data->createStream();
    
    string line;
    int last_parent = -1;
    int group_size = 0;
    int groups = 0;
    int n = 0;
    while(!stream->isEof()){
        line        = stream->readLine();
        char *id    = (char*)calloc(256, sizeof(char));
        int parent  = -1;
        sscanf(line.c_str(), "%s %d", id, &parent);
        
        t.parent    = (int*)realloc(t.parent, (n+1)*sizeof(int));
        t.parent[n] = parent;
        t.child     = (int*)realloc(t.child, (n+1)*sizeof(int));
        t.child[n]  = -1;
        t.name      = (char**)realloc(t.name, (n+1)*sizeof(char *));
        t.name[n]   = id;
        
        if(parent != last_parent){
            ++groups;
            t.group_offset              = (int*)realloc(t.group_offset, groups * sizeof(int));
            t.group_offset[groups - 1]  = n - group_size;
            t.group_size                = (int*)realloc(t.group_size, groups * sizeof(int));
            t.group_size[groups - 1]    = group_size;
            group_size                  = 0;
            last_parent                 = parent;
        }
        
        t.group     = (int*)realloc(t.group, (n+1)*sizeof(int));
        t.group[n]  = groups;
        
        if (parent >= 0) {
            t.child[parent] = groups;
        }
        ++n;
        ++group_size;
    }
    ++groups;
    t.group_offset              = (int*)realloc(t.group_offset, groups * sizeof(int));
    t.group_offset[groups - 1]  = n - group_size;
    t.group_size                = (int*)realloc(t.group_size, groups * sizeof(int));
    t.group_size[groups - 1]    = group_size;
    t.n         = n;
    t.groups    = groups;
    t.leaf      = (int*)calloc(n, sizeof(int));
    int i;
    for(i = 0; i < n; ++i) t.leaf[i] = 1;
    for(i = 0; i < n; ++i) if(t.parent[i] >= 0) t.leaf[t.parent[i]] = 0;
    
    tree *tree_ptr  = (tree*)calloc(1, sizeof(tree));
    *tree_ptr       = t;
    return tree_ptr;
}
