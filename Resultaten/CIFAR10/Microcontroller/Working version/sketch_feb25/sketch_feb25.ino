#include <avr/pgmspace.h>
#include "morepgmspace.h"
#include "Layer.h"
#include "Params.h"




const PROGMEM int8_t inputs[1728] = {3,-16,-2,3,-1,4,5,-10,4,-6,6,-13,0,6,-0,0,4,-14,5,3,-9,3,-2,6,-1,4,-2,1,1,2,2,-3,14,-4,3,-1,-1,4,2,-6,2,-8,2,0,-1,2,-0,1,-10,-8,-7,-4,-3,-5,-6,1,13,3,-5,4,4,-3,-1,-3,-3,-1,-5,-7,-6,-8,-7,-8,5,7,10,6,4,-1,-6,-7,-7,-5,-5,-4,-4,-2,0,4,4,9,6,5,4,4,3,5,4,1,-2,-2,1,-2,-4,-2,-3,-2,-2,-2,-3,-3,-3,-4,-4,0,4,5,3,2,3,2,-1,-6,4,8,6,-4,-4,-5,-6,-5,-4,-4,-5,-5,-3,-1,-0,1,1,5,4,2,2,2,-1,3,7,2,-0,-7,-1,-2,-0,-4,-3,-1,-2,-7,-2,-3,-5,-1,-1,3,5,2,2,2,3,5,1,4,6,-7,0,-1,2,-4,-2,-2,2,-4,-6,-6,-4,-2,-1,-0,4,3,1,2,4,3,-0,5,5,-5,1,-5,-4,-3,-1,-4,-3,1,-3,-2,1,1,1,4,5,3,1,1,-9,-19,-4,10,3,-7,-6,-8,-3,-1,-3,-3,-3,-2,-5,-5,-2,-5,-6,-0,4,6,4,1,-0,-9,-9,8,5,-2,0,-3,-8,-4,-2,-0,-1,1,2,-0,-1,-9,-10,-10,-9,-9,4,3,5,10,-5,-7,3,6,5,7,4,4,4,4,2,2,3,3,6,5,3,0,1,-3,-1,3,-2,5,5,3,-8,2,4,3,2,2,0,-0,3,2,1,1,1,2,2,2,3,2,1,2,3,-2,-2,0,1,3,-3,0,1,1,-0,-2,0,3,2,1,1,0,1,-0,2,2,2,2,1,-0,-0,-1,7,-5,-10,-8,6,0,-5,-1,-3,3,1,1,1,2,4,-1,-2,2,2,5,-4,-1,-2,-0,2,-6,-1,-8,2,0,0,2,-6,4,3,-1,-0,-2,-6,-1,-2,-8,-9,-6,-2,-1,-2,0,1,-3,-1,-4,-4,6,0,1,-0,4,2,-4,-3,-6,-4,-2,-4,-4,-6,-9,-1,-1,-3,1,-3,-4,-0,-2,-6,4,2,2,6,2,2,3,1,1,3,2,1,2,2,3,-2,-2,-2,-0,-6,2,1,-2,-4,-2,4,-1,-4,7,2,-0,0,2,-2,-1,-1,-4,-2,-2,-3,-2,-3,-0,-6,2,-2,-3,-3,-4,2,6,6,5,6,6,3,-2,-1,0,0,-1,-1,-2,2,2,1,3,-3,-2,5,-4,-2,-4,-1,-2,1,-4,-4,-2,5,0,2,3,3,3,3,9,-8,-4,-2,-1,-3,2,3,-2,0,-6,4,-2,-3,1,-4,-1,1,-1,1,2,2,3,4,4,-0,-7,-7,-9,-6,6,1,-1,1,-3,-2,1,1,1,2,-2,-3,-3,-2,-3,-2,-1,-2,-5,14,9,7,3,-5,1,-1,-2,1,0,-6,-7,-8,-9,-7,-7,-2,-4,-5,-7,-7,-7,-4,-1,5,-15,-1,3,-2,3,3,-11,4,-6,6,-12,-1,4,-2,-1,4,-13,5,3,-9,3,-2,4,2,7,-0,1,-1,1,0,-4,12,-6,2,-2,-4,1,-1,-7,3,-7,2,1,-0,3,1,2,-8,-6,-5,-5,-4,-4,-4,2,13,2,-5,5,3,-4,-2,-3,-3,-2,-7,-7,-5,-7,-7,-8,5,6,9,4,2,2,-1,-2,-4,-1,-1,-1,-1,1,2,4,3,6,3,3,4,3,3,4,3,-0,-4,-5,0,2,2,4,2,3,3,2,2,2,1,-2,-3,-2,0,1,1,1,1,1,-3,-7,2,5,6,2,3,-0,-1,-1,-2,-2,-1,-1,2,4,5,4,-1,1,0,1,1,1,-2,2,6,-1,-0,-1,5,-0,2,-2,-2,-1,0,-2,4,5,4,5,-0,-1,1,-0,0,0,0,3,2,4,6,-3,4,-1,3,-3,-2,-2,4,1,0,3,2,3,3,-2,-1,-0,-0,-0,3,4,0,-0,-0,-2,8,1,-2,-2,-1,-5,-3,2,-0,3,3,5,7,3,1,1,-1,-0,-8,-16,-4,2,-5,-5,2,-0,-1,0,-4,-4,-5,-4,-7,-4,-1,-3,-2,-0,2,4,3,-0,-0,-7,-11,4,4,-1,3,-2,-8,-4,-2,-1,-1,2,4,2,-1,-9,-8,-10,-9,-9,4,1,4,9,-6,-7,5,6,4,4,3,3,3,4,2,3,4,4,5,4,3,2,2,-1,-0,2,-3,2,5,4,-9,1,4,2,1,1,1,2,2,0,-1,-0,-1,1,2,3,4,3,0,-0,3,-2,-1,2,-0,2,-1,3,4,1,2,1,-0,1,-0,1,2,2,3,1,3,4,2,1,1,0,1,1,5,-7,-8,-7,8,2,-3,1,-3,1,0,2,2,3,5,-0,0,6,6,7,-4,-0,0,2,0,-9,-0,-9,3,2,2,4,-5,4,3,1,1,-1,-4,-1,-0,-4,-6,-3,-0,1,1,4,-1,-6,-0,-5,-4,7,2,2,0,3,2,-3,-1,-4,-2,-0,-3,-4,-7,-10,1,1,1,5,-3,-6,1,-3,-7,5,3,2,4,0,0,3,2,2,4,3,3,4,2,2,1,2,3,6,-6,1,2,-4,-6,0,6,-3,-7,5,-0,-2,1,3,-1,-1,1,-1,-0,-1,0,1,1,4,-7,3,1,-5,-4,0,6,5,3,4,5,4,3,-1,-1,-1,-1,-2,-2,-2,3,3,2,5,-4,-0,9,-7,-4,-1,3,-2,1,-3,-2,-3,5,-0,-1,-2,-1,1,1,9,-9,-5,-3,-2,-3,5,7,-5,-3,-5,7,-1,-2,4,0,1,3,0,-0,-1,-1,1,2,4,-3,-10,-10,-11,-7,10,7,-4,-2,-3,-1,1,1,4,5,0,-1,0,0,-1,-2,-2,-3,-5,12,7,6,2,-5,8,7,-4,-1,0,-7,-9,-9,-7,-5,-7,-3,-4,-3,-4,-5,-6,-6,-2,6,-13,-0,4,-2,1,2,-11,6,-2,5,-10,-0,4,-1,-1,4,-12,3,1,-6,5,-4,3,4,7,1,3,1,1,1,-2,11,-5,2,1,-2,2,2,-4,5,-4,3,1,1,3,1,2,-6,-5,-4,-2,-2,-3,-3,4,10,1,-4,6,4,-3,1,0,-0,0,-4,-6,-4,-4,-4,-4,6,7,9,4,2,1,-3,-4,-7,-4,-3,-2,-2,-0,2,7,5,6,3,3,5,5,4,5,3,-1,-5,-6,-1,1,-0,-0,-2,-0,0,0,0,0,-0,-1,-3,-2,0,2,1,0,1,0,-4,-8,1,3,4,1,0,-4,-2,-2,-2,-3,-3,-3,-1,1,2,2,-1,2,1,-0,-0,-0,-4,-1,4,-2,-2,-1,4,-2,3,-1,-1,0,-1,-5,0,1,-1,2,-1,0,3,1,2,1,-3,-1,-2,1,4,-3,4,-2,4,-2,-1,0,3,-2,-4,-1,-1,0,2,-2,2,2,2,1,1,2,-2,-2,-1,-2,7,-1,-2,-1,0,-4,-4,1,-2,1,1,3,5,3,3,3,1,1,-9,-14,-4,0,-5,-7,-0,-3,1,1,-4,-6,-5,-4,-8,-5,-2,-4,-4,-1,3,5,4,1,0,-4,-9,3,2,-2,2,-2,-6,-3,-3,-2,-2,1,2,1,-2,-10,-9,-10,-9,-10,4,2,5,10,-5,-7,6,7,4,5,4,4,4,4,1,2,4,4,5,4,4,2,2,-3,-1,2,-3,1,4,6,-6,4,5,3,2,1,0,2,2,0,0,1,-0,2,3,4,5,2,-0,0,4,-2,-2,1,0,3,-0,4,4,2,1,0,-1,2,1,2,2,2,2,0,2,3,2,2,1,0,-0,-0,6,-6,-8,-7,8,3,-3,1,-4,2,1,1,2,3,5,-1,-1,4,5,7,-3,-0,-1,1,1,-8,-1,-9,4,3,2,4,-6,4,3,-0,0,-1,-5,-1,-1,-6,-8,-5,0,1,1,3,-1,-5,-1,-6,-3,9,3,3,0,4,2,-4,-3,-5,-3,-1,-4,-4,-7,-10,2,1,0,4,-3,-6,-1,-5,-7,6,4,2,5,1,1,2,1,3,4,3,3,4,2,2,2,2,2,4,-6,2,1,-5,-6,0,7,-2,-6,5,1,-2,1,4,-0,0,1,-2,-0,-1,1,1,1,3,-8,4,1,-6,-7,-1,7,6,4,4,6,5,4,-1,-1,-1,0,-0,-1,-1,4,4,3,4,-5,1,9,-7,-6,-2,4,-2,1,-3,-2,-2,5,-0,0,-0,1,2,2,10,-8,-5,-3,-2,-4,6,7,-5,-4,-5,7,-1,-2,4,-0,2,3,-0,0,0,0,0,1,3,-2,-9,-9,-11,-7,11,6,-3,-2,-5,-2,1,2,4,5,0,-2,-2,-1,-2,-3,-4,-5,-6,14,9,7,2,-5,8,6,-3,-0,-2,-8,-7,-8,-8,-6,-7,-4,-6,-5,-6,-7,-8,-6,-2};
int8_t* channelpredictions;     //one channel output, used as input for pooling
int8_t* channelinput;           //one channel input for normal conv
int8_t* pooloutput;             //storage for all channels after pooling
int8_t* poolchannelpredafter;   //one channel output after pooling, will be stored in pooloutput
int8_t* poolchannelpredbefore;  //one channel output during pooling
int8_t* layerout;
int8_t* layerin;

int8_t max=0;
int8_t maxind=0;
int16_t insize = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial);
  int16_t inshape[3]= {3,24,24};
  int16_t w[4] = {32,3,3,3};
  int8_t b[1] = {32};
  int8_t conv[2] = {1,1};
  int8_t pool[3] = {2,2,0};
  Layer *layers[8];
  
  layers[0] = new Layer(inshape, w, b, true, false, sizeof(weights1), sizeof(bias1));
  layers[0]->setConv(conv);
  layers[0]->calcOutputshape();
  layers[0]->setWeights(GET_FAR_ADDRESS(weights1));
  layers[0]->setBias(GET_FAR_ADDRESS(bias1));
  
  inshape[0]=32;
  w[1]=32;
  layers[1] = new Layer(inshape, w, b, true, true, sizeof(weights2), sizeof(bias2));
  layers[1]->setConv(conv);
  layers[1]->setPool(pool);
  layers[1]->calcOutputshape();
  layers[1]->setWeights(GET_FAR_ADDRESS(weights2));
  layers[1]->setBias(GET_FAR_ADDRESS(bias2));

  
  pooloutput = new int8_t[layers[1]->getOutsize()];
  insize = sizeof(inputs)/layers[0]->inshape[0];
  channelpredictions = new int8_t[layers[0]->getOutsize()/layers[0]->outshape[0]];
  poolchannelpredbefore = new int8_t[layers[1]->getOutsize()/layers[1]->outshape[0]*(layers[1]->poolparams[0]*layers[1]->poolparams[1]/2)*(layers[1]->poolparams[0]*layers[1]->poolparams[1]/2)];

  for(int8_t opch = 0; opch<layers[1]->outshape[0]; opch++)
  {
    for(int8_t pch = 0; pch<layers[1]->inshape[0]; pch++)
    {
      channelinput = new int8_t[insize];
      for(int8_t ch=0; ch<layers[0]->inshape[0]; ch++)                                //iterate 3 input channels
      {
        for(int16_t i=0; i<insize; i++)
        {
          channelinput[i]=pgm_read_byte_near(GET_FAR_ADDRESS(inputs) + i + ch*insize);                 //get one channel
        }
        layers[0]->getOutput(channelinput, channelpredictions, ch, pch);                //get one channel output in channelpredictions
        /*for(int16_t j=0; j<layers[0]->getOutsize()/layers[0]->outshape[0]; j++)
        {
          if(j%24==0)
            Serial.println("");
          Serial.print(channelpredictions[j]);
          Serial.print(" ");
        }*/
      }
      delete channelinput;
      delete layers[0]->padinput;
      layers[0]->padinput = NULL;
      //use channelpredictions as input for poolconv, use poolchannelpred as temp storage until all 32 input channels have been used
      //when using the final inputchannel, also use a pooloutput
      if(pch<layers[1]->inshape[0]-1)
      {
        layers[1]->getOutput(channelpredictions, poolchannelpredbefore, pch, opch);        
        /*for(int16_t j=0; j<576; j++)
        {
          if(j%24==0)
            Serial.println("");
          Serial.print(poolchannelpredbefore[j]);
          Serial.print(" ");
        }   */                            
      }
      else
      {
        int16_t start = layers[1]->getOutsize()/layers[1]->outshape[0];
        poolchannelpredafter = new int8_t[start];
        layers[1]->getOutput(channelpredictions, poolchannelpredbefore, pch, opch, poolchannelpredafter);
        for(int16_t j=0; j<576; j++)
        {
          if(j%24==0)
            Serial.println("");
          Serial.print(poolchannelpredbefore[j]);
          Serial.print(" ");
        }   
        for(int16_t j=0; j<144; j++)
        {
          if(j%12==0)
            Serial.println("");
          Serial.print(poolchannelpredafter[j]);
          Serial.print(" ");
        }     
        int16_t id=0;
        for(int16_t cp=opch*start; cp<(opch+1)*start; cp++)
        {
          pooloutput[cp] = poolchannelpredafter[id];
          id++;
        }
        delete poolchannelpredafter;
      }
      delete layers[1]->padinput;
      layers[1]->padinput = NULL;
    }
  }
  delete layers[0];
  insize = layers[1]->getOutsize()/layers[1]->outshape[0];
  delete layers[1];
  delete channelpredictions;
  channelpredictions = NULL;
  delete poolchannelpredbefore;
  poolchannelpredbefore = NULL;

  Serial.println("end first pool");
    inshape[0]=32;
  inshape[1]=12;
  inshape[2]=12;
  layers[2] = new Layer(inshape, w, b, true, false, sizeof(weights3), sizeof(bias3));
  layers[2]->setConv(conv);
  layers[2]->calcOutputshape();
  layers[2]->setWeights(GET_FAR_ADDRESS(weights3));
  layers[2]->setBias(GET_FAR_ADDRESS(bias3));
  
  layers[3] = new Layer(inshape, w, b, true, true, sizeof(weights4), sizeof(bias4));
  layers[3]->setConv(conv);
  layers[3]->setPool(pool);
  layers[3]->calcOutputshape();
  layers[3]->setWeights(GET_FAR_ADDRESS(weights4));
  layers[3]->setBias(GET_FAR_ADDRESS(bias4));
  
  layerin = pooloutput;
  pooloutput = new int8_t[layers[3]->getOutsize()];
  channelpredictions = new int8_t[layers[2]->getOutsize()/layers[2]->outshape[0]];
  poolchannelpredbefore = new int8_t[layers[3]->getOutsize()/layers[3]->outshape[0]*(layers[3]->poolparams[0]*layers[3]->poolparams[1]/2)*(layers[3]->poolparams[0]*layers[3]->poolparams[1]/2)];
  for(int8_t opch = 0; opch<layers[3]->outshape[0]; opch++)
  {
    for(int8_t pch = 0; pch<layers[3]->inshape[0]; pch++)
    {
      channelinput = new int8_t[insize];
      for(int8_t ch=0; ch<layers[2]->inshape[0]; ch++)                                //iterate 32 input channels
      {
        for(int16_t i=0; i<insize; i++)
        {
          channelinput[i]=layerin[i + ch*insize];                 //get one channel
        }
        layers[2]->getOutput(channelinput, channelpredictions, ch, pch);                //get one channel output in channelpredictions
        /*for(int16_t j=0; j<layers[2]->getOutsize()/layers[2]->outshape[0]; j++)
        {
          if(j%12==0)
            Serial.println("");
          Serial.print(channelpredictions[j]);
          Serial.print(" ");
        }*/
      }
      delete channelinput;
      delete layers[2]->padinput;
      layers[2]->padinput = NULL;
      //use channelpredictions as input for poolconv, use poolchannelpred as temp storage until all 32 input channels have been used
      //when using the final inputchannel, also use a pooloutput
      if(pch<layers[3]->inshape[0]-1)
      {
        layers[3]->getOutput(channelpredictions, poolchannelpredbefore, pch, opch);        
        /*for(int16_t j=0; j<144; j++)
        {
          if(j%12==0)
            Serial.println("");
          Serial.print(poolchannelpredbefore[j]);
          Serial.print(" ");
        }   */                            
      }
      else
      {
        int16_t start = layers[3]->getOutsize()/layers[3]->outshape[0];
        poolchannelpredafter = new int8_t[start];
       layers[3]->getOutput(channelpredictions, poolchannelpredbefore, pch, opch, poolchannelpredafter);
        /*for(int16_t j=0; j<144; j++)
        {
          if(j%12==0)
            Serial.println("");
          Serial.print(poolchannelpredbefore[j]);
          Serial.print(" ");
        }  */ 
        for(int16_t j=0; j<36; j++)
        {
          if(j%6==0)
            Serial.println("");
          Serial.print(poolchannelpredafter[j]);
          Serial.print(" ");
        }     
        int16_t id=0;
        for(int16_t cp=opch*start; cp<(opch+1)*start; cp++)
        {
          pooloutput[cp] = poolchannelpredafter[id];
          id++;
        }
        delete poolchannelpredafter;
      }
      delete layers[3]->padinput;
      layers[3]->padinput = NULL;
    }
  }
  delete layerin;
  
  delete layers[2];
  delete layers[3];
  delete channelpredictions;
  channelpredictions = NULL;
  delete poolchannelpredbefore;
  poolchannelpredbefore = NULL;
  
  Serial.println("end second pool");
  inshape[0]=32;
  inshape[1]=6;
  inshape[2]=6;
  w[0]=64;
  b[0]=64;
  layers[4] = new Layer(inshape, w, b, true, false, sizeof(weights5), sizeof(bias5));
  layers[4]->setConv(conv);
  layers[4]->calcOutputshape();
  layers[4]->setWeights(GET_FAR_ADDRESS(weights5));
  layers[4]->setBias(GET_FAR_ADDRESS(bias5));
 
  layerin = pooloutput;
  layerout = new int8_t[layers[4]->getOutsize()];
  layers[4]->getOutput(layerin, layerout);
  delete layers[4];
  delete layerin;

  for(int16_t j=0; j<2304; j++)
  {
    if(j%6==0)
      Serial.println("");
    if(j%36==0)
      Serial.println("");
    Serial.print(layerout[j]);
    Serial.print(" ");
  }  

  inshape[0]=64;
  w[1]=64;
  layers[5] = new Layer(inshape, w, b, true, true, sizeof(weights6a)+sizeof(weights6b), sizeof(bias6));
  layers[5]->setConv(conv);
  layers[5]->setPool(pool);
  layers[5]->calcOutputshape();
  layers[5]->setWeights(GET_FAR_ADDRESS(weights6a));
  layers[5]->setBias(GET_FAR_ADDRESS(bias6));
  layers[5]->extraweights = GET_FAR_ADDRESS(weights6b);
  layers[5]->weightoffset = sizeof(weights6a);
  
  
  layerin = layerout;
  layerout = new int8_t[layers[5]->getOutsize()];

  int16_t outsize = 1;
  for(int8_t i=0; i<3; i++)
  {
    outsize*=layers[5]->outshape[i];
  }
  pooloutput = new int8_t[outsize];
  channelinput = new int8_t[36];

  for(int16_t ich=0; ich<layers[5]->inshape[0];ich++)
  {
    for(int8_t i=0; i<36; i++)
    {
      channelinput[i]=layerin[ich*36+i];  
      if(ich==60)
      {
        if(i%6==0)
          Serial.println("");
        Serial.print(channelinput[i]);
        Serial.print(" ");
        Serial.print(layerin[ich*36+i]);
        Serial.print(" (");
        Serial.print(ich*36+i);
        Serial.print(") ");
      }
    }  
    layers[5]->getOutput(channelinput, pooloutput, ich, -1, layerout);
  }
  
  delete layers[5];
  delete layerin;
  delete pooloutput;
  delete channelinput;

  for(int16_t j=0; j<576; j++)
  {
    if(j%3==0)
      Serial.println("");
    if(j%9==0)
      Serial.println("");
    Serial.print(layerout[j]);
    Serial.print(" ");
  }   

  Serial.println("end third pool");
  inshape[0]=64;
  inshape[1]=3;
  inshape[2]=3;
  w[0]=576;
  w[1]=64;
  b[0]=64;
  layers[6] = new Layer(inshape, w, b, false, false, sizeof(weights7a)+sizeof(weights7b), sizeof(bias7));
  layers[6]->calcOutputshape();
  layers[6]->setWeights(GET_FAR_ADDRESS(weights7a));
  layers[6]->setBias(GET_FAR_ADDRESS(bias7));
  layers[6]->extraweights = GET_FAR_ADDRESS(weights7b);
  layers[6]->weightoffset = sizeof(weights7a);

  layerin = layerout;
  layerout = new int8_t[layers[6]->getOutsize()];
  layers[6]->getOutput(layerin, layerout);
  delete layers[6];
  delete layerin;

  for(int16_t j=0; j<64; j++)
  {
    Serial.print(layerout[j]);
    Serial.print(" ");
  } 

  inshape[0]=64;
  inshape[1]=0;
  inshape[2]=0;
  w[0]=64;
  w[1]=10;
  b[0]=10;
  layers[7] = new Layer(inshape, w, b, false, false, sizeof(weights8), sizeof(bias8));
  layers[7]->calcOutputshape();
  layers[7]->setWeights(GET_FAR_ADDRESS(weights8));
  layers[7]->setBias(GET_FAR_ADDRESS(bias8));

  layerin = layerout;
  layerout = new int8_t[layers[7]->getOutsize()];
  layers[7]->getOutput(layerin, layerout);
  delete layers[7];
  delete layerin;
  Serial.println("");
  for(int8_t i=0; i<10; i++)
  {
    Serial.print(layerout[i]);Serial.print(" ");  
  }
 
  Serial.println("Final cleanup");
  delay(1000); 
  delete layerout;
  Serial.println("End Prediction");
}

void loop() {
  // put your main code here, to run repeatedly:

}
