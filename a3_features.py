import argparse
import pandas as pd
from glob import glob
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

# Whatever other imports you need

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20",
                        help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()

    print("Reading {}...".format(args.inputdir))
    # Do what you need to read the documents here.
    # read file from path
    counts = {}
    i = 0
    text_list = []
    author_list = []
    folders = glob("{}/*".format(args.inputdir))
    for author in folders:
        author_name = author[13:]
        path = glob("{}/*".format(author))
        for file in path:
            text = ''
            with open(file, "r") as thefile:
                for line in thefile:
                    text += line

            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            text_list.append(text)
            author_list.append(author_name)
            counts[author_name] = {i: text}
            i += 1

    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
    # vectorize
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(text_list)

    # dimensionality reduction
    svd = TruncatedSVD(n_components=args.dims)
    X = svd.fit_transform(x)
    y = author_list
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.testsize / 100)

    # Build the table here.
    table = pd.DataFrame()

    y_train = pd.DataFrame(y_train)
    y_train['train_test'] = 'train'
    y_train.columns = ['author', 'train_test']
    y_test = pd.DataFrame(y_test)
    y_test['train_test'] = 'test'
    y_test.columns = ['author', 'train_test']
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    
    table_y = pd.concat([y_train, y_test])
    table_x = pd.concat([X_train, X_test])
    table = pd.concat([table_y, table_x], axis=1)
    print("Writing to {}...".format(args.outputfile))
    # Write the table out here.
    table.to_csv(args.outputfile)
    print("Done!")
