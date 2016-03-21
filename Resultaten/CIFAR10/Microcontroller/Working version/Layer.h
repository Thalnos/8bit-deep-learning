#ifndef LAYER_H
#define LAYER_H
//#include <limits.h>

#define SCHAR_MAX 1023
#define SCHAR_MIN -1024
class Layer
{
	public:
		int16_t inshape[3];						//specific 3D-inputs and outputs 
		int16_t outshape[3];
		int8_t * weights;						//weights
		int8_t * bias;							//bias
		bool conv;								//conv or dense	
		int8_t convparams[3];					//extra params for conv: stride and pad, filtersize(square)
		int8_t poolparams[3];					//extra params when layer is followed by pool: size(square), stride, pad
		bool pool;								//followed by pool or not
		long weightsize;						//size of weights array
		int16_t biassize;						//size of bias array
		int8_t * padinput;
		int16_t* partialweightsizes;
		int16_t partialweight;

		Layer();
		Layer(int16_t[3], int16_t*, int8_t*, bool, bool, long, int16_t);
		~Layer();
		void getOutput(const int8_t *, int8_t*, int8_t, int8_t, int8_t*);			//calculate output with given input
		void calcOutputshape();
		void setWeights(int8_t*);
		void setBias(int8_t*);
		void setConv(int8_t[2]);
		void setPool(int8_t[3]);
		void createWeights(int);				//set weight size
		void createBias(int);					//set bias size
		int16_t getOutsize();
		void writeAll();
};

// Member functions definitions including constructor
Layer::Layer()
{
	conv = false;
	pool = false;
	for(int8_t i=0;i<3;i++)
	{
		inshape[i]=0;
		outshape[i]=0;
		convparams[i]=0;
		poolparams[i]=0;
	}
	padinput = NULL;
	partialweightsizes=NULL;
	partialweight=0;
	weights = NULL;
	bias = NULL;
	weightsize=0;
	biassize=0;
}
Layer::Layer(int16_t in[3], int16_t*w, int8_t*b, bool c, bool p, long wsize, int16_t bsize)
{
  conv = c;
  pool = p;
  weightsize = wsize;
  biassize = bsize;
  for(int8_t i=0; i<3; i++)
  {
    inshape[i] = in[i];
  }
  if(c)
  {
    outshape[0] = w[0];
    convparams[2] = w[2];
  }
  else
  {
    outshape[0] = w[1];
  }
}
Layer::~Layer()
{
	if(weights!=NULL)
		delete weights;

	if(bias!=NULL)	
		delete bias;

	if(padinput!=NULL)
		delete padinput;	
}
void Layer::writeAll()
{
	std::cout << "in\t" << "out" << std::endl;
	for(int i=0; i<3; i++)
	{
		std::cout << inshape[i] << "\t" << outshape[i] << std::endl;
	}
	std::cout << "convlayer: " << conv << std::endl;
	std::cout << "pool: " << pool << std::endl;
	std::cout << "conv\t" << "pool" << std::endl;
	for(int i=0; i<3; i++)
	{
		std::cout << convparams[i] << "\t" << poolparams[i] << std::endl;
	}
}
void Layer::createWeights(int w)
{
	weights = new int8_t[w];
	weightsize = w;
}
void Layer::createBias(int b)
{
	bias = new int8_t[b];
	biassize = b;
}
void Layer::setConv(int8_t cparams[2])
{
  convparams[0] = cparams[0];
  convparams[1] = cparams[1];
}
void Layer::setPool(int8_t pparams[3])
{
  poolparams[0] = pparams[0];
  poolparams[1] = pparams[1];
  poolparams[2] = pparams[2];
}
void Layer::setWeights(int8_t*w)
{
  weights = w;
}
void Layer::setBias(int8_t*b)
{
  bias = b;
}

void Layer::calcOutputshape()
{
	if(conv)
	{
		outshape[1] = inshape[1]-(convparams[2]-1-2*convparams[1]);
		outshape[2] = inshape[2]-(convparams[2]-1-2*convparams[1]);			//stride not implemented 
	}
}
int16_t Layer::getOutsize()
{
  if(conv)
  {
    int16_t outsize = 1;
    for(int8_t i=0; i<3; i++)
    {
		if(i>0 && pool)
		{
			outsize*=outshape[i]/(poolparams[0]*poolparams[1]/2);
		}
		else
		{
			outsize*=outshape[i];
		}
    }
    return outsize;
  }
  else
  {
      return outshape[0];
  }
}
void Layer::getOutput(const int8_t * input, int8_t*output, int8_t in = -1, int8_t out = -1, int8_t* pooloutput=NULL)
{
	int16_t insize = 1;
	int16_t padinsize = 1;
	int16_t outsize = 1;
	int16_t y = 0;
	int16_t poolsize = 1;

	for(int8_t i=0; i<3; i++)
	{
		insize*=inshape[i];
	}
	if(in!=-1)
	{
		insize/=inshape[0];
	}
	for(int8_t i=0; i<3; i++)
	{
		if(i>0)
		{
			padinsize*=inshape[i]+convparams[1]*2;
		}
		else
		{
			padinsize*=inshape[i];
		}	
	}
	poolsize = getOutsize();
	for(int8_t i=0; i<3; i++)
	{
		outsize*=outshape[i];
	}
	
	if(conv)				//Layer is Convlayer
	{
		
		if(padinput==NULL)
		{
			padinput = new int8_t[padinsize];
		}
		if(pool && out==-1 && in==-1)
		{
			pooloutput = output;
			output = new int8_t[outsize];
		}
		int16_t count = 0;
		int16_t count2 = 0;
		if(in==-1)
		{
			for(int i=0; i<outsize; i++)
			{
				output[i]=0;
			}
		}
		
//		std::cout << "padding" << std::endl;
		int16_t channelstop = inshape[0];
		if(in!=-1)
		{
			channelstop = 1;
			count2 = in*(inshape[2]*inshape[1]);
		}
		for(int16_t u=0; u<channelstop; u++)
		{
			for(int8_t o=0; o<inshape[1]+convparams[1]*2; o++)
			{
				for(int8_t l=0; l<inshape[2]+convparams[1]*2; l++)
				{
					if(o<inshape[1]+convparams[1] && o>convparams[1]-1 && l<inshape[2]+convparams[1] && l>convparams[1]-1)
					{
						padinput[count] = input[count2];
						if(in==60 && out==-1)
						{
							if(count2%6==0)
							std::cout << std::endl;
							int16_t wr = input[count2];
							std::cout << wr << " ";//" (" << count << ") ";
						}
						count2++;
					}
					else
					{
						padinput[count] = 0.0;
					}
					/*if(in==60 && out==-1)
					{
						if(count%8==0)
							std::cout << std::endl;
						int16_t wr = padinput[count];
						std::cout << wr << " ";//" (" << count << ") ";
					}*/
					count++;
				}	
			/*	if(u==0)
				{
					std::cout << std::endl;
				}	*/	
			}
		}	
		

//		std::cout << std::endl;
		int16_t outchannelstop = outshape[0];
		if(out!=-1)
		{
			outchannelstop = 1;
		}
		for(int8_t i=0; i<outchannelstop; i++)
		{
			int16_t wi = i;
			if(out!=-1)
			{
				wi = out;
			}
				
			for(int16_t j=0; j<outsize/outshape[0]; j++)
			{
				int16_t part = 0;
				if(in!=-1)
				{
					channelstop = 1;
				}
				
				for(int16_t u=0; u<channelstop; u++)
				{
					for(int8_t o=0; o<convparams[2]; o++)
					{
						for(int8_t l=0; l<convparams[2]; l++)
						{
							int16_t wu = u;
							if(in!=-1)
							{
								wu = in;
							}
							y = padinput[u*((inshape[2]+convparams[1]*2)*(inshape[1]+convparams[1]*2)) + (j/inshape[2])*(inshape[2]+convparams[1]*2) + j%inshape[2] + o*(inshape[2]+convparams[1]*2) + l] * weights[wi*(weightsize/outshape[0]) + wu*(convparams[2]*convparams[2]) + ((convparams[2]*convparams[2])-1 - (o*convparams[2] + l))];						
							if(i==63 && in==60 && out==-1 && j==0)
							{
								int16_t wei = weights[wi*(weightsize/outshape[0]) + wu*(convparams[2]*convparams[2]) + ((convparams[2]*convparams[2])-1 - (o*convparams[2] + l))];
								int16_t padin = padinput[u*((inshape[2]+convparams[1]*2)*(inshape[1]+convparams[1]*2)) + (j/inshape[2])*(inshape[2]+convparams[1]*2) + j%inshape[2] + o*(inshape[2]+convparams[1]*2) + l];
								int16_t write = output[i*(outshape[1]*outshape[2]) + j];
								
								std::cout << "padindex: " << u*((inshape[2]+convparams[1]*2)*(inshape[1]+convparams[1]*2)) + (j/inshape[2])*(inshape[2]+convparams[1]*2) + j%inshape[2] + o*(inshape[2]+convparams[1]*2) + l << "  padin: " << padin << "  *  weight: " << wei << "   windex: " << wi*(weightsize/outshape[0]) + wu*(convparams[2]*convparams[2]) + ((convparams[2]*convparams[2])-1 - (o*convparams[2] + l));
								std::cout << "ind: " << i*(outshape[1]*outshape[2]) + j << " => " << y << std::endl;
							}
									

//TODO saturate overflow here or not?
		
							/*if (y > 0 && output[i*(outshape[1]*outshape[2]) + j] > SCHAR_MAX - y)
							{
								output[i*(outshape[1]*outshape[2]) + j] = SCHAR_MAX;
							}
							else if (y < 0 && output[i*(outshape[1]*outshape[2]) + j] < SCHAR_MIN - y)
							{
								output[i*(outshape[1]*outshape[2]) + j] = SCHAR_MIN;
							}
							else
							{*/
								part += y;
								//output[i*(outshape[1]*outshape[2]) + j] += y;
							//}				
						}
					}
				}
				
				if(i==0 && in==63 && out==-1)
				{
					int16_t write = part;
					if(j%6==0)
						std::cout << std::endl;
					std::cout << write << " ";
						
				}
				if(in==-1 || in==inshape[0]-1)
				{
					if(biassize==outsize)
					{
						y = bias[wi*(outshape[1]*outshape[2]) + j];
						/*if(i==0)
							std::cout << "bias: " << y;*/
					}
					else
					{
						y = bias[wi];
					}
					for(int8_t k=0; k<8; k++)
					{
						part+=y;
						//output[i*(outshape[1]*outshape[2]) + j]+=y;
					}
				}
				if(i==0 && in==63 && out==-1)
				{
					int16_t write = part;
					if(j%6==0)
						std::cout << std::endl;	
					std::cout << write << " ";
					
				}

				if(in==-1 || in==inshape[0]-1)
				{
					int8_t add = 4;
					if(in==inshape[0]-1)
					{
						/*if(i==0 && (in==-1 || in == inshape[0]-1)  && (out==-1 || out == 1))
						{
							int16_t st = output[i*(outshape[1]*outshape[2]) + j];
							std::cout << "stuff: " << st <<  " (" << j << ") ";
						}*/
						part += output[i*(outshape[1]*outshape[2]) + j]*8;					//rounding down using separate input channels isn't completely correct, this could be fixed by saving 16bit output channels
					/*	if(i==0 && (in==-1 || in == inshape[0]-1)  && (out==-1 || out == 1))
						{
							std::cout << "part: " << part <<  " ";
						}*/
					}	
					if(part < 0)
					{
						if (part < SCHAR_MIN + add)
						{
							part = SCHAR_MIN;
						}
						else
						{
							part-=add;	
						}	
					}
					else
					{	
						if (part > SCHAR_MAX - add)
						{
							part = SCHAR_MAX;
						}
						else
						{
							part+=add;	
						}	
					}	
				}
				part/=8; 
				if(i==0 && in==63 && out==-1)
				{
					int16_t write = part;
					if(j%6==0)
						std::cout << std::endl;	
					std::cout << write << " ";	
				}


				if(in==-1 || in==inshape[0]-1)
				{
					if(part < 0)
					{
						output[i*(outshape[1]*outshape[2]) + j] = 0;			
					}
					else
					{
						output[i*(outshape[1]*outshape[2]) + j] = part;
					}
				}
				else
				{
					if(in==0)
					{
						output[i*(outshape[1]*outshape[2]) + j] = part;
					}
					else if(in>0 && in<inshape[0]-1)
					{
						if(part<0)
						{
						/*	if(i==0 && (in==-1 || in <= 2)  && (out==-1 || out == 0))
							{	
								int16_t owr = output[i*(outshape[1]*outshape[2]) + j];
								std::cout << "part: " << part << " out: " << owr << " ";
							}*/
							if(output[i*(outshape[1]*outshape[2]) + j] < -127 - part)
							{
								output[i*(outshape[1]*outshape[2]) + j] = -127;
							}
							else
							{
								output[i*(outshape[1]*outshape[2]) + j] += part;
							}
						}
						else
						{
						/*	if(i==0 && (in==-1 || in <= 2)  && (out==-1 || out == 0))
							{	
								int16_t owr = output[i*(outshape[1]*outshape[2]) + j];
								std::cout << "part: " << part << " out: " << owr << " ";
							}*/
							if(output[i*(outshape[1]*outshape[2]) + j] > 126 - part)
							{
								output[i*(outshape[1]*outshape[2]) + j] = 126;
							}
							else
							{
								output[i*(outshape[1]*outshape[2]) + j] += part;
							}
						}
					}
				}

				if(i==0 && in==63 && out==-1)
				{
					int16_t write = output[i*(outshape[1]*outshape[2]) + j];
					if(j%6==0)
						std::cout << std::endl;	
					std::cout << write << " ";
				}
			}
		/*	if(pool && i==1)
			{
				std::cout << "started new channel" << std::endl;
				for(int16_t j=0; j<12; j++)
				{
					for(int16_t k=0; k<12; k++)
					{
						int16_t ind = 0*144 + j*12 + k;
						int16_t write = pooloutput[ind];
						std::cout << write << " ";
						if(ind%11==0 && ind!=0)
							std::cout << std::endl;
					}
				}
			}	*/
			if(pool && (in==-1 || in == inshape[0]-1))
			{
				for(int16_t j=0; j<outshape[1]/(poolparams[0]*poolparams[1]/2); j++)
				{
					for(int16_t k=0; k<outshape[2]/(poolparams[0]*poolparams[1]/2); k++)
					{
						pooloutput[i*outshape[1]/(poolparams[0]*poolparams[1]/2)*outshape[2]/(poolparams[0]*poolparams[1]/2) + j*outshape[1]/(poolparams[0]*poolparams[1]/2) + k] = output[i*(outshape[1]*outshape[2]) + j*(poolparams[0]*poolparams[1]/2)*outshape[1] + k*(poolparams[0]*poolparams[1]/2)];
						for(int16_t l = j*(poolparams[0]*poolparams[1]/2); l<j*(poolparams[0]*poolparams[1]/2)+poolparams[0]; l++)
						{
							for(int16_t m = k*(poolparams[0]*poolparams[1]/2); m<k*(poolparams[0]*poolparams[1]/2)+poolparams[0]; m++)
							{
								if(output[i*(outshape[1]*outshape[2]) + l*outshape[1] + m]>pooloutput[i*outshape[1]/(poolparams[0]*poolparams[1]/2)*outshape[2]/(poolparams[0]*poolparams[1]/2) + j*outshape[1]/(poolparams[0]*poolparams[1]/2) + k])
								{
									pooloutput[i*outshape[1]/(poolparams[0]*poolparams[1]/2)*outshape[2]/(poolparams[0]*poolparams[1]/2) + j*outshape[1]/(poolparams[0]*poolparams[1]/2) + k] = output[i*(outshape[1]*outshape[2]) + l*outshape[1] + m];
								}
							/*	if(i==2 && j==2)
								{
									std::cout << "out: " << l*outshape[1] + m << " => " << output[i*(outshape[1]*outshape[2]) + l*outshape[1] + m] << " ";
									std::cout << "poolind: " << j*outshape[1]/(poolparams[0]*poolparams[1]/2) + k << " => " <<  pooloutput[i*outshape[1]/(poolparams[0]*poolparams[1]/2)*outshape[2]/(poolparams[0]*poolparams[1]/2) + j*outshape[1]/(poolparams[0]*poolparams[1]/2) + k] << "||" << output[i*(outshape[1]*outshape[2]) + l*outshape[1] + m] << "&&";
								}*/
							}
						}
					/*	if(i==0)
						{
							int16_t ind = i*144 + j*12 + k;
							int16_t write = pooloutput[ind];
							std::cout << write << " ";
							if(ind%12==0 && ind!=0)
								std::cout << std::endl;	
						}*/
					}
				}
			}
		}
		if(pool && (in==-1 || in == inshape[0]-1) && (out==-1 || out == 0))
		{
			std::cout << "\npool" << std::endl;
			for(int16_t j=0; j<3; j++)
			{
				for(int16_t k=0; k<3; k++)
				{
					int16_t ind = 0*9 + j*3 + k;
					int16_t write = pooloutput[ind];
					std::cout << write << " ";
				}
				std::cout << std::endl;
			}
		}
		if(pool && in==-1 && out==-1)
		{
			delete output;
			output = pooloutput;
		}
	}
	else					//Layer is Denselayer
	{
		insize = 1;
		for(int8_t i=0; i<3; i++)
		{
			if(inshape[i]>0)
			{
				insize*=inshape[i];
			}
		}
		for(int16_t i=0; i<outshape[0]; i++)
		{
			int16_t part = 0;
			for(int16_t j=0; j<insize ; j++)
			{
				y = input[j]*weights[i + j*outshape[0]];
				if(i==1)
				{				
					std::cout << "windex: " << i + j*outshape[0] << " w: " << weights[i + j*outshape[0]]*1;
					std::cout << " in: " << input[j]*1 << std::endl;
				}
				if (y > 0 && part > SCHAR_MAX - y)
				{
					part = SCHAR_MAX;
				}
				else if (y < 0 && part < SCHAR_MIN - y)
				{
					part = SCHAR_MIN;
				}
				else
				{
					part += y;
				}	
			}

			if(i==1)
			{
				std::cout << part << " ";
			}

			if(part < 0)
			{
				part-=4;
				if(part>=0)
				{
					part=SCHAR_MIN;
				}       
			}
			else
			{
				part+=4;
				if(part<0)
				{
					part=SCHAR_MAX;
				}       
			}
			part/=8;

			if(i==1)
			{
				std::cout << part << " ";
			}

			y = bias[i];
			if (y > 0 && part > SCHAR_MAX - y)
			{
				part = SCHAR_MAX;
			}
			else if (y < 0 && part < SCHAR_MIN - y)
			{
				part = SCHAR_MIN;
			}
			else
			{
				part += y;
			} 

			if(i==1)
			{
				std::cout << part << " ";
			}

			if(part>0 || outshape[0]==10)
			{				
				output[i]=part;	
			}
			else
			{
				output[i]=0;
			}
//			std::cout << output[i]*1 << " ";
		}	
	}
	
}

#endif