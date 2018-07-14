using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    using matrix;
    using ActivationFunctions;

    namespace Layers
    {
        using NeuralNetwork;

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

        /// <summary>
        /// 默认的神经层的缩小，使用单个数
        /// </summary>
        public class SmallDenes : Layer<Matrix, float>
        {
            Random r = new Random();
            public SmallDenes(string Name, AF af = null) : base(Name, af)
            {
                Weight = (float)r.NextDouble();
                Bias = 0;

            }
            public override Matrix Function(Matrix input)
            {
                Matrix ou = input;
                matrix.Function.Multiply(ref ou, Weight);
                matrix.Function.Add(ref ou, Bias);
                return ou;
            }

            public override Matrix FunctionNoAF(Matrix input)
            {
                Matrix ou = input;
                matrix.Function.Multiply(ref ou, Weight);
                matrix.Function.Add(ref ou, Bias);
                return ou;
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
    /// 神经网络
    /// </summary>
    /// <typeparam name="Tensor">输出输出数据的类型,现只支持Matrix</typeparam>
    /// <typeparam name="LayerWB">神经网络中权重和偏置的类型,现只支持Matrix</typeparam>
    public class Network<Tensor, LayerWB> where Tensor : Matrix where LayerWB : Matrix
    {
        private Dictionary<string, Layer<Tensor, LayerWB>> Structure;

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
        /// <param name="isShow">是否显示过程</param>
        public void Train(Tensor XData, Tensor YData, bool isShow = true)
        {
            List<Tensor> tensors = new List<Tensor>() { XData };
            string link = "Input";
            for (int i = 0; i < Structure.Count; i++)
            {
                Layer<Tensor, LayerWB> layer = Structure[link];
                tensors.Add(layer.Function(tensors[i]));
                link = layer.LinkLayerName;
            }
            Matrix Out = tensors[tensors.Count - 1];
            Function.Sub(ref Out, YData);
            Function.Power(ref Out, 2);
            float loss = Function.Mean(Out);
            Console.WriteLine("loss:" + loss);


            Matrix pian = tensors[tensors.Count - 1];
            Function.Sub(ref pian, YData);
            Function.Multiply(ref pian, 2);
            
            for (int i = Structure.Count-2; i >= 0; i--)
            {
                Layer<Tensor, LayerWB> thislayaer = FindLayerBynum(i);
                if (thislayaer.Name == "Input")
                    break;
                Layer<Tensor, LayerWB> lastlayaer = FindLayerBynum(i-1);

                Tensor Dlayer = tensors[i-1];
                Tensor Dzl = Dlayer;
                Dlayer = thislayaer.FunctionNoAF(Dlayer);
                if (thislayaer.ActivationFunction != null)
                    Dlayer.Derivatives(FindLayerBynum(i).ActivationFunction);

                Matrix offset_w = pian;
                Function.Multiply(ref offset_w, Dlayer);
                Matrix offset_b = offset_w;
                float of_b = Function.AddAll(offset_b);
                Matrix offset_ll = offset_b;
                Function.Multiply(ref offset_w, Dzl);
                Matrix w = thislayaer.Weight;
                Matrix b = thislayaer.Bias;
                Matrix ll = tensors[i-1];
                offset_ll = w * offset_ll;
                float of_w = Function.AddAll(offset_w);
                float of_ll = Function.AddAll(offset_ll);

                Function.Add(ref w, of_w);
                Function.Add(ref b, of_b);
                Function.Add(ref ll, of_ll);
                thislayaer.Weight = (LayerWB)w;
                thislayaer.Bias = (LayerWB)b;
                Function.Multiply(ref ll, 2);
                pian = ll;
            }

        }

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
