#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <stdint.h>

#include "Layer.h"



int main(void)
{
	FILE * fp;
	char * line = NULL;
	size_t len = 0;
	ssize_t read;
	char * partline;
	const char * delim = ",";
	char* weights1[600];
	char* bias1[8];
	char* weights2[46080];
	char* bias2[10];
	int count = 0;
	int count2 = 0;
	int samplenr = 0;
	int8_t inputs[100][1728];
	char* target[100];
//	int16_t padinput[2352];
	int16_t output1[4608];
	int16_t totalsum[100][10];
	int prediction[100];
	int16_t max = 0;
	int maxind = 0;
	double accuracy = 0;
	bool weights = false;
	int layernr = 0;
	int out = 0;
	int8_t y = 0;
	Layer *layers[8];
	int dim;
	int8_t param = 0;

	fp = fopen("paramsmc_lim.csv", "r");
	if (fp == NULL)
		exit(EXIT_FAILURE);

	read = getline(&line, &len, fp);
	while (read != -1) 
	{
		if(strstr(line, "new layer") != NULL)
		{
			std::cout << "New layer" << std::endl;
			read = getline(&line, &len, fp);
			layers[layernr] = new Layer();
			while(read != -1 && strstr(line, "new layer") == NULL)				//read properties until values are read
			{
				if(strstr(line, "w") != NULL)									//weights
				{
					weights = true;
					dim = 1;
					int8_t tel=0;
					while ((partline = strsep(&line, delim)) != NULL)
					{
						if(std::isdigit(partline[0]))
						{
							dim*=atoi(partline);
							if(layers[layernr]->conv)
							{
								if(tel==1)
								{
									layers[layernr]->outshape[0]=dim;
								}
								else if(tel==3)
								{
									layers[layernr]->convparams[2]=atoi(partline);
								}
							}
							else
							{
								std::cout << "dense" <<std::endl;
								if(tel==2)
								{
									layers[layernr]->outshape[0]=atoi(partline);			//denselayer output is 1D
								}
							}
						}
						tel++;
					}
					layers[layernr]->createWeights(dim);
				}
				else if(strstr(line, "b") != NULL)								//bias
				{
					weights = false;
					dim = 1;
					while ((partline = strsep(&line, delim)) != NULL)
					{
						if(std::isdigit(partline[0]))
						{
							dim*=atoi(partline);
						}
					}
					layers[layernr]->createBias(dim);
				}
				else if(strstr(line, "conv") != NULL)							//convlayer props
				{
					param = 0;
					while ((partline = strsep(&line, delim)) != NULL)
					{
						if(std::isdigit(partline[0]))
						{
							layers[layernr]->convparams[param] = atoi(partline);
							param++;
						}
					}
					layers[layernr]->conv = true;
				}
				else if(strstr(line, "Pool") != NULL)							//conv is followed by a pool layer
				{
					param = 0;
					while ((partline = strsep(&line, delim)) != NULL)
					{
						if(std::isdigit(partline[0]))
						{
							layers[layernr]->poolparams[param] = atoi(partline);
							param++;
						}
					}
					layers[layernr]->pool = true;
				}
				else if(strstr(line, "params") != NULL)
				{
					count = 0;
					read = getline(&line, &len, fp);
					while(read != -1 && strstr(line, "end params") == NULL)		//read values
					{
						while ((partline = strsep(&line, delim)) != NULL)
						{
							if(strstr(partline, "\n")==NULL)								
							{
								if(weights)
								{
									if(count<layers[layernr]->weightsize)
									{												
										layers[layernr]->weights[count] = atoi(partline);
									}
								}
								else
								{
									if(count<layers[layernr]->biassize)
									{	
										layers[layernr]->bias[count] = atoi(partline);
									}
								}		
								count++;
							}
						}
						
						read = getline(&line, &len, fp);
					}	
					layernr+=!weights;
					std::cout << "end params" << std::endl;
				}
				else															//inputshape
				{
					param = 0;
					while ((partline = strsep(&line, delim)) != NULL)
					{
						if(std::isdigit(partline[0]))
						{
							layers[layernr]->inshape[param] = atoi(partline);
							param++;
						}
					}
				}
				read = getline(&line, &len, fp);
			}
			
		}
	}

	fclose(fp);

	std::cout << "Read input" << std::endl;
	fp = fopen("inputmc_lim.csv", "r");
	if (fp == NULL)
		exit(EXIT_FAILURE);
	count = 0;
	samplenr = 0;
	while ((read = getline(&line, &len, fp)) != -1) 
	{
//		printf("Retrieved line of length %zu :\n", read);
//		printf("%s", line);
		while ((partline = strsep(&line, delim)) != NULL)
  		{
//    		printf("%s  ", partline);
			if(strstr(partline, "\n") == NULL)
			{
				inputs[samplenr][count] = atoi(partline);		
				count++;
				count%=1728;
			}
			
  		}
		samplenr++;
//		std::cout << "weights: " << weights1 << std::endl;
	}

	std::cout << "Read targets" << std::endl;
	fp = fopen("targetmc_lim.csv", "r");
	if (fp == NULL)
		exit(EXIT_FAILURE);
	count = 0;
	samplenr = 0;
	while ((read = getline(&line, &len, fp)) != -1) 
	{
//		printf("Retrieved line of length %zu :\n", read);
//		printf("%s", line);
		while ((partline = strsep(&line, delim)) != NULL)
  		{
//    		printf("%s  ", partline);
			if(strstr(partline, "\n") == NULL)
			{
				target[samplenr] = partline;	
//				std::cout << "id: " << samplenr << std::endl;
				samplenr++;	
			}
			
  		}
//		std::cout << "weights: " << weights1 << std::endl;
	}

/*	for(uint i=0; i<100; i++)
	{
		for(uint j=0; j<sizeof(inputs[i])/sizeof(inputs[i][0]); j++)
		{
			std::cout << inputs[i][j] << " ";
		}
		std::cout << std::endl << std::endl;
	}
*/	
	
	if (line)
		free(line);
	if (partline)
		free(partline);
	
	for(int i=0; i<8; i++)
	{
		layers[i]->calcOutputshape();
		layers[i]->writeAll();				
	}
	int8_t *imagein;
	int8_t *predictions;
	int8_t *poolpredictions;
	int8_t *channelpredictions;
	int8_t *poolchannelpred;
	int8_t *channelin;
	int8_t *poolout;
	for(int k=0; k<1; k++)
	{
		imagein = inputs[k];
		std::cout << std::endl;
		for(int i=0; i<8; i++)
		{
//			std::cout << "Layer: " << i+1 << std::endl;
//			std::cout << "Outsize: " << layers[i]->getOutsize() << std::endl;
			layers[i]->partialweight = 0;
			
			if(layers[i]->pool && i<=3)
			{
				channelpredictions = new int8_t[layers[i]->getOutsize()/layers[i]->outshape[0]*(layers[i]->poolparams[0]*layers[i]->poolparams[1]/2)*(layers[i]->poolparams[0]*layers[i]->poolparams[1]/2)];
				poolchannelpred = new int8_t[layers[i]->getOutsize()/layers[i]->outshape[0]];
				predictions = new int8_t[layers[i]->getOutsize()*(layers[i]->poolparams[0]*layers[i]->poolparams[1]/2)*(layers[i]->poolparams[0]*layers[i]->poolparams[1]/2)];
				poolpredictions = new int8_t[layers[i]->getOutsize()];			
			}
			else if(i==5)
			{
				channelin = new int8_t[36];
				predictions = new int8_t[layers[i]->getOutsize()];
				poolout = new int8_t[2304];
			}
			else
			{
				channelpredictions = new int8_t[layers[i]->getOutsize()/layers[i]->outshape[0]];
				predictions = new int8_t[layers[i]->getOutsize()];
			}
			if(layers[i]->conv && i<=3)
			{
				for(int j=0; j<layers[i]->inshape[0]; j++)
				{			
					for(int k=0; k<layers[i]->outshape[0]; k++)
					{
						int16_t id = 0;
						int16_t start = layers[i]->getOutsize()/layers[i]->outshape[0];
						if(layers[i]->pool)
						{
							start*=(layers[i]->poolparams[0]*layers[i]->poolparams[1]/2)*(layers[i]->poolparams[0]*layers[i]->poolparams[1]/2);
						}
						if(j>0)
						{
							id=0;
							for(int l=k*start; l<(k+1)*start; l++)
							{
								channelpredictions[id] = predictions[l];
								id++;
							}
						}
						if(layers[i]->pool && j==layers[i]->inshape[0]-1)
							layers[i]->getOutput(imagein, channelpredictions, j, k, poolchannelpred);
						else
							layers[i]->getOutput(imagein, channelpredictions, j, k);			
						id=0;
						if(layers[i]->pool && j==layers[i]->inshape[0]-1)
						{
							start = layers[i]->getOutsize()/layers[i]->outshape[0];
						}
						for(int l=k*start; l<(k+1)*start; l++)
						{	
							if(layers[i]->pool && j==layers[i]->inshape[0]-1)
							{
								poolpredictions[l] = poolchannelpred[id];
							}	
							else
							{
								predictions[l] = channelpredictions[id];
							}
							id++;
						}	
					}		
				}
			}
			else if(i==5)
			{
				std::cout << "INPUT" << std::endl;
				/*for(int p=0; p<2304;p++)
				{
					if(p%6==0)
						std::cout << std::endl;
					if(p%36==0)
						std::cout << std::endl;
					std::cout << imagein[p]*1 << " ";
					
				}*/
				for(int ich=0; ich<64; ich++)
				{
					for(int m=0; m<36; m++)
					{
						channelin[m]=imagein[ich*36+m];
						if(ich==60)
						{
							if(m%6==0)
								std::cout << std::endl;
							std::cout << channelin[m]*1 << " ";
						}
					}
					layers[i]->getOutput(imagein, poolout, ich, -1, predictions);
				}
				for(int p=0; p<576;p++)
				{
					if(p%3==0)
						std::cout << std::endl;
					if(p%9==0)
						std::cout << std::endl;
					std::cout << predictions[p]*1 << " ";
				}
			}
			else
			{
				layers[i]->getOutput(imagein, predictions);
				std::cout << "OUTPUT" << std::endl;
				/*for(int p=0; p<2304;p++)
				{
					if(p%6==0)
						std::cout << std::endl;
					if(p%36==0)
						std::cout << std::endl;
					std::cout << predictions[p]*1 << " ";
					
				}*/
				if(i==6)
				{
				for(int i=0; i<64; i++)
				{
					std::cout << predictions[i]*1 << " ";
				}
				}
			}
	//		std::cout << "\nall channels";
	//		layers[i]->getOutput(imagein, predictions);
			
			/*if(i==1)
			{
				std::cout << "predictions" << std::endl;
				for(int j=0; j<12; j++)
				{
					for(int k=0; k<12; k++)
					{
						int16_t write = predictions[0 * 144 + j*12 + k];
						std::cout << write << " ";
					}
					std::cout << std::endl;
				}
			}
			*/
			if(i>0)
				delete imagein;
			if(layers[i]->pool && i<=3)
			{
				imagein = poolpredictions;
				delete poolchannelpred;
			}
			else if(i==5)
			{
				imagein = predictions;
				delete poolout;
			}
			else
				imagein = predictions;
			delete channelpredictions;
//			std::cout << "End layer: " << i+1 << std::endl;
		}
		int16_t write;
		std::cout << "pred: ";
		max = 0;
		for(int8_t i=0; i<10; i++)
		{
			write = predictions[i];
			std::cout << write << " ";
			if(write>max)
			{
				max = write;
				maxind = i;
			}
		}
		std::cout << std::endl;
		delete predictions;
		prediction[k]=maxind;
	}
	
	for(int8_t i=0; i<8; i++)
	{
		delete(layers[i]);
	}

	for(int k=0; k<100; k++)
	{
	//	std::cout << "prediction: " << prediction[k] << "  expected: " << target[k] << std::endl;
		if(prediction[k]==atoi(target[k]))
		{
			accuracy++;		
		}
	}
	std::cout << "accuracy: " << accuracy/100.0 << "%" << std::endl;

	


	return 0;
}