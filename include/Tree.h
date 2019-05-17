//
//  Tree.h
//  StreetViewAnalyzer
//
//  Created by HAOZHE LI on 08/04/2019.
//

#ifndef Tree_h
#define Tree_h

namespace ml {
    typedef struct{
        int *leaf;
        int n;
        int *parent;
        int *child;
        int *group;
        char **name;
        
        int groups;
        int *group_size;
        int *group_offset;
    } tree;
    
    float get_hierarchy_probability (float *x, tree *hier, int c, int stride);
    void  hierarchy_predictions     (double *predictions, int n, tree *hier, int only_leaves, int stride);
    int   hierarchy_top_prediction  (double *predictions, tree *hier, float thresh, int stride);
    tree  *read_tree(ci::DataSourceRef data);
};

#endif /* Tree_h */
