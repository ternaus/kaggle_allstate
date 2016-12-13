#include <cmath>
#include <string.h>
#include <iostream>
#include <iomanip>
#include <deque>
#include <vector>
#include <map>
#include <algorithm>
#include <fstream>


using namespace std;
//Helper Functions
void ReplaceStringInPlace(string& subject, const string& search,
                          const string& replace) {
    size_t pos = 0;
    while ((pos = subject.find(search, pos)) != string::npos) {
         subject.replace(pos, search.length(), replace);
         pos += replace.length();
    }
}

vector<double> tokenize(const char* src,
                                  char delim,
                                  bool want_empty_tokens)
{
  vector<double> tokens;

  if (src and *src != '\0') // defensive
    while( true )  {
      const char* d = strchr(src, delim);
      size_t len = (d)? d-src : strlen(src);

      if (len or want_empty_tokens)
      {
         string s = string(src, len);
         tokens.push_back(stod(s)  );// capture token
      }

      if (d) src += len+1; else break;
    }

  return tokens;
}

vector<string> tokenizeString(const char* src,
                                  char delim,
                                  bool want_empty_tokens)
{
  vector<string> tokens;

  if (src and *src != '\0') // defensive
    while( true )  {
      const char* d = strchr(src, delim);
      size_t len = (d)? d-src : strlen(src);

      if (len or want_empty_tokens)
      {
         string s = string(src, len);
         s.erase(s.find_last_not_of(" \n\r\t")+1);
         tokens.push_back(s);// capture token
      }

      if (d) src += len+1; else break;
    }

  return tokens;
}

inline bool sortByDecreasingPrediction(pair<double,double>x,pair<double,double>y)
{
    return x.first > y.first;
}

inline bool sortByIncreasingPrediction(pair<double,double>x,pair<double,double>y)
{
    return x.first < y.first;
}

void Regression(vector<pair<double,double>>&data, double min=0, double max=1000000)
{
    cout << "Starting Regression"<<endl;
    data.push_back(pair<double,double>(min, min));
    data.push_back(pair<double,double>(max, max));

    int sizeOfData = data.size();
    if(sizeOfData<=1)
        return;

    cout << "Starting Sorting"<<endl;
    sort(data.begin(),data.end(),sortByIncreasingPrediction);
    cout << "Stopped Sorting"<<endl;

    vector<double> solution;
    solution.reserve(sizeOfData);
    for(int i=0;i<sizeOfData;i++)
    {
        solution[i]=data[i].second;
    }

    for(int i=1;i<sizeOfData-1;i++)
    {
        if(i%100000==0)
        {
             cout << endl << i/100000 << endl;
        }
        else if(i%1000==0)
        {
            cout << ".";
            cout.flush();
        }

        double denominator = 0.0;
        double numerator = 0.0;
        if(solution[i]>solution[i+1])
        {
            //We need to average
            numerator+=solution[i];
            denominator++;
            numerator+=solution[i+1];
            denominator++;


            int j=0;
            while(((i-j-1)>=0) and (solution[i-j-1]>(numerator/denominator)))
            {
                numerator+=solution[i-j-1];
                denominator++;
                j++;

            }

            while(j>0)
            {
                solution[i-j]=numerator/denominator;
                j--;
            }

            solution[i]=numerator/denominator;
            solution[i+1]=numerator/denominator;
        }

    }

    for(int i=0;i<sizeOfData;i++)
    {
        data[i].second=solution[i];
    }
    solution.clear();

}


double Predict(vector<pair<double,double>>&data, double input )
{

    //Important it is assumed that the data is monotonic!!

    auto len = std::distance(data.begin(), data.end());
    vector<pair<double,double>>::iterator begin = data.begin();
    vector<pair<double,double>>::iterator end = data.end();

    //Boundary Conditions
    if(input<begin->second)
        return begin->second;

    if(input>(end-1)->second)
        return (end-1)->second;

    //Binary Chop
    while(len>1)
    {
        vector<pair<double,double>>::iterator middleElem = begin;
        std::advance(middleElem, len / 2);

        if (middleElem->first == input)
        {
            return middleElem->second;

        }
        else if (middleElem->first < input)
        {


            begin = middleElem;

        }
        else
        {

            end = middleElem;


        }
        len = std::distance(begin, end);
    }


    return begin->second+((end->second-begin->second)/(end->first-begin->first))*(input-begin->first);

}

vector<pair<double,double>> LoadTrainingData(string trainingFile)
{
    cout << endl << "Load Training Data" << endl;
    bool useHeader =true;
    vector<string> header;
    string line;
    ifstream datastream;
    datastream.open(trainingFile);
    vector<pair<double,double>> data;
    if(datastream.good())
    {
        int noOfEntries = 0;
        while(getline(datastream, line)&&line.length()>0)
        {

            vector<string> lineData  = tokenizeString(line.c_str(),',',false);
            if(useHeader)
            {
                header=lineData;
                useHeader=false;
            }
            else
            {
                int column = 0;
                double predicted = 0;
                double actual = 0;
                for(auto &entry:lineData)
                {
                    if(header[column]=="id")
                    {
                        //Ignore Id
                    }
                    else if(header[column]=="prediction")
                    {
                        predicted= stod(entry);
                        if(std::isnan(predicted))
                        {
                            cout << "Error:" << header[column] << endl;
                            exit(0);
                        }
                        
                    }
                    else if(header[column]=="loss")
                    {
                        actual= stod(entry);
                        if(std::isnan(actual))
                        {
                            cout << "Error:" << header[column] << endl;
                            exit(0);
                        }
                    }

                    column++;
                 }

                 pair<double,double> a(log(predicted+200.0),log(actual+200.0));
                 data.push_back(a);
            }

            if(noOfEntries%1000000==0)
            {
                cout << ".";
                cout.flush();
            }
            noOfEntries++;

        }
    }

    datastream.close();
    return data;
}

bool Test(string path, string submission, vector<pair<double,double>>&data, bool addactual=true)
{
    cout << endl << "Writing Testing Data" << endl;
    ofstream out(submission);
    if(!out)
    {
        cout << "Cannot open submission file." << endl;
        return false;
    }
    vector<string> header;//Column Names
    string line;
    ifstream datastream(path);
    if(datastream.good())
    {
        bool isHeader = true;
        int noOfEntries = 0;
        while(getline(datastream, line)&&line.length()>0)
        {
             if(isHeader)
             {
                header=tokenizeString(line.c_str(),',',false);
                isHeader=false;
                if(addactual)
                {
                    out << "id,prediction,loss" <<endl;
                }
                else
                {
                    out << "id,loss" << endl;
                }


             }
             else
             {

                if(noOfEntries%1000000==0)
                {
                    cout << endl << "Written "<< noOfEntries << " Test Rows" << endl;
                }
                else if(noOfEntries%100000==0)
                {
                    cout << ".";
                    cout.flush();
                }

                vector<string> lineData  = tokenizeString(line.c_str(),',',false);
                string id = "";
                int column = 0;
                double prediction=0.00;
                double actual=0.00;

                for(auto &x : lineData)
                {

                    if(header[column]=="id")
                    {
                       id=x;
                    }
                    else if(header[column]=="prediction")
                    {
                       prediction=stod(x);
                    }
                    else if(header[column]=="loss")
                    {
                       actual=stod(x);

                    }
                    
                    column++;
                }
                

                double p = exp(Predict(data, log(prediction+200.0)))-200.0;

                if(addactual)
                {
                    out << id << "," << std::setprecision(4) << p << "," << actual  << endl;
                }
                else
                {
                    out << id << "," << std::setprecision(4) << p << endl;

                }

             }

            noOfEntries++;
        }

        datastream.close();
    }
    else
    {
       out.close();
       cout << endl << "Couldn't open file: " << path <<endl;
       return false;
    }

    return true;
}

int main()
{
    string directory = "/home/vladimir/workspace/kaggle_allstate/src/calibration/";
    //Expects id, prediction, loss as columns 
    string trainFile = directory + "trainsubmission.csv";
    string regressiontrainFile = directory + "isotrain.csv";
    //Expects id, prediction as columns
    string testFile = directory + "testsubmission.csv";
    string regressiontestFile = directory + "isotest.csv";
    cout << std::setprecision(4) ;
    vector<pair<double,double>>data = LoadTrainingData(trainFile);//It will be sorted
    unsigned int noOfRows = data.size();
    cout << endl << "No Of Rows:" << noOfRows << endl;
    Regression(data, log(200.0), log(120200.0));
    Test(trainFile, regressiontrainFile, data, true);
    Test(testFile, regressiontestFile, data, false);
    return 0;
}
