import xml.etree.ElementTree as ET
import string
import spacy
import re
import logging
import os

class Patent(object) :
    """
    Class to handle reading information from a patent record .xml file
    or an xml tree.
    ============================

    Parameters:
    -----------

    tree        xml tree root or a filename of .xml file containing instructions
                for constructing such tree

    Methods:
    --------
    get_node            returns the first node (preorder search) whose tag
                        matches the one provided
    get_all_nodes       returns all nodes whose tag matches the one provided
                        as a list (preorder search)

    get_description     returns a dictionary containing information about the 
                        Description part of the patent.
    get_claims          returns a dictionary containing information about the 
                        Claims part of the patent.
    get_all_text        returns all text elements in the 'patent tree'
    lemmatize_tree      lemmatize all text elements. 
    print_tree          neatly print all tags in the tree


    TO DO: 
    - get_independent claims
    - get_language


    """
    def __init__(self, tree) :
        if type(tree) == str :
            self._filename = os.path.basename(tree)
            self._root = ET.parse(tree).getroot()
        elif isinstance(tree, ET.ElementTree) :
            self._root = tree.getroot()
        else :
            print("ERROR: 'tree' must be xml.etree.ElementTree.ElementTree "
                  "instance or a filename")
            self._root = None
            

    @staticmethod
    def get_node(root, tag) :
        """
        Traverse tree until node.tag == tag
        return node

        """

        if root == None :
            return None

        all_nodes = Patent.get_all_nodes(root, tag)
        if len(all_nodes) == 0:
            return None
        else :
            return all_nodes[0]


    @staticmethod
    def get_all_nodes(root, tag) : 
        """
        return all nodes (as a list) with node.tag == tag in the tree

        use with get_node(sub-tree, tag) to reduce search to a specific sub-tree
        """
        if root is None :
            return None
        return list(root.iter(tag))

    def get_text_node(self, tag) :
        nd = Patent.get_node(self._root, tag)
        if nd is not None :
            tx = nd.text
            if tx is not None :
                return tx
        logging.debug(f"could not find text in tag {tag}")
        return ""

    def pprint(self) :
        " Pretty print patent xml tree"
        print("Title:\n\t", self.get_title())
        print("Descrioption:\n\t", self.get_description())
        claims = self.get_claims()
        for c in claims :
            print(c,":\n\t",claims[c])

    def lemmatize_tree(self, lemmatizer) :
        """
        Replace all text elements in the tree by their tokenized version
        """

        def travel(node) :
            if node.text :
                node.text = lemmatizer(node.text) 
            for l in list(node) :
                travel(l)
        
        travel(self._root)

    def to_xml(self, fn) :
        with open(fn, 'w') as outfile :
            outfile.write(ET.tostring(self._root, encoding="unicode"))

    def add_element(self, tag, text) :
        ne = ET.Element(tag)
        ne.text = text
        self._root.append(ne)

    def get_title(self) :
        rt = Patent.get_node(self._root, 'technical-data')
        if rt == None :
            rt = self._root
        all_titles = Patent.get_all_nodes(rt, 'invention-title')
        return " ".join([tl.text.strip() for tl in all_titles if tl.text != None])

    def print_tree(self) :
        def travel(node, depth) :
            print("".join(['|--']*depth), node.tag, end=' ')
            if node.text : 
                if len(node.text.strip()) >0 :
                    print(' = ', node.text)
                else :
                    print(" ")
            for l in list(node) :
                travel(l, depth+1)
        travel(self._root, 0)

    def get_classification_code(self) :
        """
        CPC classification
        """
        rt = list(Patent.get_all_nodes(self._root, 'classification-cpc'))
        return [tl.text for tl in rt]

    def get_id(self) :
        return Patent.get_node(self._root, 'doc-number').text

    def get_description(self) :    
        """ 
        return the 'description' part of a patent
        if 'description' can't be found, return 'disclosure' part
        if 'disclosure' can't be found, return 'abstract' 

        """
        description = Patent.get_node(self._root, 'description') 
        if description != None :
            ps = Patent.get_all_nodes(description, 'p')
            return " ".join([p.text for p in ps if p.text])

        disclosure = Patent.get_node(self._root, 'disclosure')
        if disclosure != None :
            ps = Patent.get_all_nodes(disclosure, 'p')
            return " ".join([p.text for p in ps if p.text])

        abstract = Patent.get_node(self._root, 'abstract')
        if abstract != None :
            ps = Patent.get_all_nodes(abstract, 'p')
            return " ".join([p.text for p in ps if p.text])

        #logging.warning(f"Patent::{self.get_id()}: Did not find desctiption/disclosure/abstract.")
        return ""
    
    def get_claims(self) :
        """
        Get Claims part; arrange in a dictionary 
        {'claim1' : <text of claim1>, ... }
        """
        claims = Patent.get_all_nodes(Patent.get_node(self._root, 'claims'),'claim') 
        ls = {}
        if claims == None :
            ls['claim1'] = ""
        else :
            for i,c in enumerate(claims) :
                ls['{}{}'.format(c.tag, i+1)] = " ".join(c.itertext())
        return ls
    
    def get_length(self) :
        return len(self.get_all_text().split())
    
    def get_all_text(self) :
        """
        All text elements in the patent tree
        """
        return Patent.get_tree_text(self._root)
    
    def get_rejections(self) :
        rej = Patent.get_all_nodes(self._root,'Rejection')
        return [r.text for r in rej]

    @staticmethod
    def get_tree_text(tree) :
        """Travers trought tree (root first) and return all text elements of leafs"""
        ls = list(tree)
        if len(ls) == 0 :
            if tree.text == None :
                return ""
            else :
                return tree.text
        else :
            dt = ""
            for e in ls :
                dt += Patent.get_tree_text(e) + ' '
            return dt
