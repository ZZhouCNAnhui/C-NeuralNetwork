using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{

    using Layers;
    using matrix;
    using ActivationFunctions;

    namespace Layers
    {

        public abstract class Layer<IO, WB>
        {
            public Layer(string Name, AF af)
            {
                this.Name = Name;
                this.ActivationFunction = af;
            }
            public int LayerNumber;
            public AF ActivationFunction;
            public WB Weight;
            public WB Bias;
            public abstract IO Function(IO input);
            public abstract IO FunctionNoAF(IO input);
            public string Name;
            public string LinkLayerName;

        }

        public class Constant<T1, T2> : Layer<T1, T2>
        {
            public Constant(string Name) : base(Name, null) { }

            public override T1 Function(T1 input)
            {
                return input;
            }

            public override T1 FunctionNoAF(T1 input)
            {
                return input;
            }
        }

        /// <summary>
        /// 默认的神经层，使用矩阵
        /// </summary>
        public class Denes : Layer<Matrix, Matrix>
        {
            Random r = new Random();
            /// <summary>
            /// 构造神经层
            /// </summary>
            /// <param name="inputsize">输入数据大小</param>
            /// <param name="outputsize">输出数据大小</param>
            /// <param name="Name">神经层名字</param>
            /// <param name="af">激励函数</param>
            /// <param name="WeightRa">是否权重随机</param>
            public Denes(int inputsize, int outputsize, string Name, AF af = null, bool WeightRa = true) :
                base(Name, af)
            {

                if (WeightRa)
                    Weight = matrix.Function.Random(inputsize, outputsize, -1, 1);
                else
                    Weight = new Matrix(inputsize, outputsize, 2);
                Bias = new Matrix(1, outputsize, 0);
            }
            public override Matrix Function(Matrix input)
            {
                Matrix M = Weight * input + Bias;
                if (ActivationFunction != null)
                    M.Activation(ActivationFunction);
                return M;
            }

            public override Matrix FunctionNoAF(Matrix input)
            {
                Matrix M = Weight * input + Bias;
                return M;
            }
        }
        

    }

    /// <summary>
    /// 激励函数
    /// </summary>
    namespace ActivationFunctions
    {
        /// <summary>
        /// 激励函数类型
        /// </summary>
        /// <param name="input">输出数据</param>
        /// <param name="Derivatives">是否为导函数</param>
        /// <returns>输出结果</returns>
        public delegate float AF(float input, bool Derivatives);
        public static class ActivationFunctions
        {
            public static AF Relu = delegate (float input, bool Derivatives)
            {
                if (Derivatives)
                {
                    if (input > 0)
                        return 1;
                    else
                        return 0;
                }
                else
                {
                    return Math.Max(0, input);
                }
            };

            public static AF Sigmoid = delegate (float input, bool Derivatives)
            {
                if (Derivatives)
                {
                    return (float)((1 / (1 + Math.Pow(Math.E, -input))) * (1 - (1 / (1 + Math.Pow(Math.E, -input)))));
                }
                else
                {
                    return (float)(1 / (1 + Math.Pow(Math.E, -input)));
                }
            };
        }
    }



    /// <summary>
    /// 神经网络
    /// </summary>
    /// <typeparam name="Tensor">输出输出数据的类型,现只支持Matrix</typeparam>
    /// <typeparam name="LayerWB">神经网络中权重和偏置的类型,现只支持Matrix</typeparam>
    public class Network<Tensor, LayerWB> where Tensor : Matrix where LayerWB : Matrix
    {
        public Dictionary<string, Layer<Tensor, LayerWB>> Structure;

        private Constant<Tensor, LayerWB> Input;
        private Constant<Tensor, LayerWB> Output;

        public Network()
        {
            Input = new Constant<Tensor, LayerWB>("Input");
            Output = new Constant<Tensor, LayerWB>("Output");
            Structure = new Dictionary<string, Layer<Tensor, LayerWB>>
            {
                { "Input", Input },
                { "Output", Output }
            };

        }

        /// <summary>
        /// 在神经网络中添加一个层
        /// </summary>
        /// <param name="layer">继承自Layer的一个层</param>
        /// <param name="LinkLayer">此层输出传递给的层的名字</param>
        public void AddLayer(Layer<Tensor, LayerWB> layer, string LinkLayer)
        {
            Structure.Add(layer.Name, layer);
            Structure[layer.Name].LinkLayerName = LinkLayer;
            string link = "Input";
            for (int i = 0; i < Structure.Count - 1; i++)
            {
                Layer<Tensor, LayerWB> la = Structure[link];
                la.LayerNumber = i;
                link = la.LinkLayerName;
            }
            Output.LayerNumber = Structure.Count-1;
        }

        /// <summary>
        /// 设置Tensor输入传递给那一层
        /// </summary>
        /// <param name="LinkLayer">Tensor输入传递给的层的名字</param>
        public void SetInputLink(string LinkLayer)
        {
            Input.LinkLayerName = LinkLayer;
        }

        /// <summary>
        /// 运行神经网络
        /// </summary>
        /// <param name="T">输入数据</param>
        /// <returns>输出Output层的输出</returns>
        public Tensor RunNetwork(Tensor T)
        {
            Tensor data = T;
            string link = "Input";
            for (int i = 0; i < Structure.Count; i++)
            {
                Layer<Tensor, LayerWB> layer = Structure[link];
                data = layer.Function(data);
                link = layer.LinkLayerName;
            }
            return data;
        }
        public List<Tensor> RunNetwork_s(Tensor T)
        {
            List<Tensor> datas = new List<Tensor>() { T};
            string link = "Input";
            for (int i = 0; i < Structure.Count; i++)
            {
                Layer<Tensor, LayerWB> layer = Structure[link];
                datas.Add(layer.Function(datas[i]));
                link = layer.LinkLayerName;
            }
            datas.Remove(T);

            return datas;
        }


        public Tensor RunForVariable(Tensor T, int LayerNumber)
        {
            Tensor data = T;
            string link = "Input";
            for (int i = 0; i < Structure.Count; i++)
            {
                Layer<Tensor, LayerWB> layer = Structure[link];
                data = layer.Function(data);
                if (layer.LayerNumber == LayerNumber)
                    return data;
                link = layer.LinkLayerName;
            }
            return data;
        }
        public Layer<Tensor, LayerWB> FindLayerBynum(int LayerNumber)
        {
            string link = "Input";
            for (int i = 0; i < Structure.Count; i++)
            {
                Layer<Tensor, LayerWB> la = Structure[link];
                if (la.LayerNumber == LayerNumber)
                    return la;
                link = la.LinkLayerName;
            }
            return null;
        }

        /// <summary>
        /// 训练神经网络
        /// </summary>
        /// <param name="XData">输入数据</param>
        /// <param name="YData">输出数据</param>
        /// <param name="learn">学习效率</param>
        /// <param name="isShow">是否显示误差</param>
        public void GradientDescent(Tensor XData, Tensor YData, float learn=0.001f,bool isShow = true)
        {
            var outputs = RunNetwork_s(XData);
            var dloss = (outputs[outputs.Count - 1] - YData) * 2;
            if (isShow)
                Console.WriteLine("[loss]:"+Function.Mean(Function.Power(outputs[outputs.Count-1]-YData,2)));
            for (int i = Structure.Count - 2; i >= 0; i--)
            {
                var thislayer = FindLayerBynum(i);
                if (thislayer.Name == "Input")
                    break;
                var Lastlayer = FindLayerBynum(i-1);

                Matrix thislayerW = thislayer.Weight;
                Matrix thislayerB = thislayer.Bias;
                Matrix lastlayerOut = outputs[i - 1];
                Matrix thislayerlz = thislayer.FunctionNoAF((Tensor)lastlayerOut);
                if (thislayer.ActivationFunction != null)
                    thislayerlz.Derivatives(thislayer.ActivationFunction);
                else
                    thislayerlz.Foreach(a => { return 1; });

                Matrix dloss_m = new Matrix(lastlayerOut.xLength, thislayerW.xLength,0);
                for (int k = 0; k < thislayerW.xLength; k++)
                {
                    for (int j = 0; j < thislayerW.yLength; j++)
                    {
                        //float wjk = thislayerW.vMatrix[k, j];
                        //float bj = thislayerB.vMatrix[0, j];
                        Matrix d_cost = dloss[-1, j];
                        Matrix d_layer = lastlayerOut[-1, k];
                        Matrix d_zl = thislayerlz[-1, j];
                        //Console.WriteLine("d_cost:" + d_cost.Shape);
                        //Console.WriteLine("d_layer:" + d_layer.Shape);
                        //Console.WriteLine("d_zl:" + d_zl.Shape);


                        Matrix offset_w_M = Function.Multiply(d_cost, d_layer);
                        offset_w_M = Function.Multiply(offset_w_M, d_zl);
                        float offset_w = Function.AddAll(offset_w_M) * learn;
                        //thislayerW.vMatrix[k, j] -= offset_w;

                        Matrix offset_b_M = Function.Multiply(d_cost, d_zl);
                        float offset_b = Function.AddAll(offset_b_M) * learn;

                        Matrix offset_ll_M = Function.Multiply(d_cost, d_zl);
                        offset_ll_M = Function.Multiply(offset_ll_M, thislayerW.vMatrix[k, j]);
                        Matrix offset_ll = offset_b_M * learn * 2;

                        thislayerW.vMatrix[k, j] -= offset_w;
                        thislayerB.vMatrix[0,j] -= offset_b;
                        dloss_m[-1,k] = offset_ll;
                    }
                }


                //thislayerW -= (dloss * lastlayerOut *thislayerlz) * learn;
                //thislayerB -= (dloss * thislayerlz) * learn;
                //dloss = dloss * thislayerlz * thislayer.Weight * learn*2;

                thislayer.Weight = (LayerWB)thislayerW;
                thislayer.Bias = (LayerWB)thislayerB;
                dloss = dloss_m;
            }

        }

        #region
        //public void Train_fortest(Tensor XData, Tensor YData)
        //{
        //    var outputs = RunNetwork_s(XData);
        //    Matrix dloss = (outputs[outputs.Count-1] - YData) * 2;
        //   // Console.WriteLine(Function.Power(outputs[outputs.Count - 1] - YData,2));
        //    Layer<Tensor, LayerWB> l1 = FindLayerBynum(1);
        //    Layer<Tensor, LayerWB> l2 = FindLayerBynum(2);

        //    Matrix l1w = l1.Weight;
        //    Matrix l1b = l1.Bias;
        //    Matrix l2w = l2.Weight;
        //    Matrix l2b = l2.Bias;
        //    Matrix dloss_l1 =  ( l1w * dloss)*0.001f;

        //    l2w -= (dloss * outputs[outputs.Count - 3]) * 0.001f;
        //    l2b -= dloss * 0.001f;
        //    l1w -= (dloss_l1 * XData) * 0.001f;
        //    l1b -= dloss_l1 * 0.001f;

        //    l2.Weight = (LayerWB)l2w;
        //    l2.Bias = (LayerWB)l2b;
        //    l1.Weight = (LayerWB)l1w;
        //    l1.Bias = (LayerWB)l1b;
        //}
        #endregion

        public override string ToString()
        {
            string s = "";
            string link = "Input";
            for (int i = 0; i < Structure.Count; i++)
            {
                Layer<Tensor, LayerWB> la = Structure[link];
                s+= string.Format("LayerName:{0}\tLayerNum:{1}\tLinkLayer:{2}",la.Name,la.LayerNumber,la.LinkLayerName);
                s += "\n";
                link = la.LinkLayerName;
            }
            return s;
        }
    }

}
