using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CntkTitanic2
{
    class Program
    {
        static void Main(string[] args)
        {
            // Input values: Pclass,Sex,Age,Siblings/Spouses Aboard,Parents/Children Aboard,Fare
            var inputSize = 6;
            // Output values: Survived 0: no, 1: yes
            var outputSize = 1;
            int[] layers = new int[] { inputSize, 12, 12, 12, outputSize };
            const int batchSize = 50;
            const int epochCount = 250;

            var dc = new Data.DataCollector();
            var training_data = dc.ReadData(@"InputData\titanic_training.csv");
            List<float> training_inputs = new List<float>();
            List<float> training_outputs = new List<float>();
            int trainingCount = training_data.Count;

            var testing_data = dc.ReadData(@"InputData\titanic_testing.csv");
            List<float> testing_inputs = new List<float>();
            List<float> testing_outputs = new List<float>();
            int testingCount = testing_data.Count;

            var dp = new Data.DataProcessor();
            dp.ProcessData(training_data, training_inputs, training_outputs);
            dp.ProcessData(testing_data, testing_inputs, testing_outputs);

            Variable x;
            Function y;
            // Build graph
            x = Variable.InputVariable(new int[] { layers[0] }, DataType.Float, "x");

            Function lastLayer = x;
            for (int i = 0; i < layers.Length - 1; i++)
            {
                Parameter weight = new Parameter(new int[] { layers[i + 1], layers[i] }, DataType.Float, CNTKLib.GlorotNormalInitializer());
                Parameter bias = new Parameter(new int[] { layers[i + 1] }, DataType.Float, CNTKLib.GlorotNormalInitializer());

                Function times = CNTKLib.Times(weight, lastLayer);
                Function plus = CNTKLib.Plus(times, bias);
                lastLayer = CNTKLib.Sigmoid(plus);
            }

            y = lastLayer;

            Variable yt = Variable.InputVariable(new int[] { outputSize }, DataType.Float);
            Function loss = CNTKLib.BinaryCrossEntropy(y, yt);

            Function y_rounded = CNTKLib.Round(y);
            Function y_yt_equal = CNTKLib.Equal(y_rounded, yt);

            Learner learner = CNTKLib.SGDLearner(new ParameterVector(y.Parameters().ToArray()), new TrainingParameterScheduleDouble(1.0, batchSize));
            Trainer trainer = Trainer.CreateTrainer(y, loss, y_yt_equal, new List<Learner>() { learner });

            for (int epochI = 0; epochI <= epochCount; epochI++)
            {
                double sumLoss = 0;
                double sumEval = 0;

                for (int batchI = 0; batchI < trainingCount / batchSize; batchI++)
                {
                    Value x_value = Value.CreateBatch(x.Shape, training_inputs.GetRange(batchI * batchSize * inputSize, batchSize * inputSize), DeviceDescriptor.CPUDevice);
                    Value yt_value = Value.CreateBatch(yt.Shape, training_outputs.GetRange(batchI * batchSize * outputSize, batchSize * outputSize), DeviceDescriptor.CPUDevice);
                    var inputDataMap = new Dictionary<Variable, Value>()
                    {
                        { x, x_value },
                        { yt, yt_value }
                    };

                    trainer.TrainMinibatch(inputDataMap, false, DeviceDescriptor.CPUDevice);
                    sumLoss += trainer.PreviousMinibatchLossAverage() * trainer.PreviousMinibatchSampleCount();
                    sumEval += trainer.PreviousMinibatchEvaluationAverage() * trainer.PreviousMinibatchSampleCount();
                }
                Console.WriteLine(String.Format("{0}\tOverall loss:{1}\tOverall accuracy:{2}", epochI, sumLoss / trainingCount, sumEval / trainingCount));
            }

            Console.WriteLine("Evaluating training inputs.");
            var trainingEval = Evaluate(inputSize, outputSize, batchSize, training_inputs, training_outputs, trainingCount, x, y);
            Console.WriteLine($"Training evaluation average: {trainingEval}");
            
            Console.WriteLine("Evaluating testing inputs.");
            var testingEval = Evaluate(inputSize, outputSize, batchSize, testing_inputs, testing_outputs, testingCount, x, y);
            Console.WriteLine($"Testing evaluation average: {testingEval}");

            EvaluateSingle(inputSize, x, y);

            Console.ReadLine();
        }

        private static void EvaluateSingle(int inputSize, Variable x, Function y)
        {
            float[] x_store = new float[inputSize];

            // 0,3,Mr. Stoytcho Mionoff,male,28,0,0,7.8958
            x_store[0] = 3f / 10;
            x_store[1] = 0f;
            x_store[2] = 28f / 100;
            x_store[3] = 0f;
            x_store[4] = 0f;
            x_store[5] = 7.8958f / 1000;
            var expectedOutput = 0;

            ////1,3,Miss.Katherine Gilnagh, female,16,0,0,7.7333
            //x_store[0] = 3f / 10;
            //x_store[1] = 1f;
            //x_store[2] = 16f / 100;
            //x_store[3] = 0f;
            //x_store[4] = 0f;
            //x_store[5] = 7.7333f / 1000;
            //var expectedOutput = 1;

            var inputDataMap3 = new Dictionary<Variable, Value>() { { x, Value.CreateBatch(x.Shape, x_store, DeviceDescriptor.CPUDevice) } };
            var outputDataMap = new Dictionary<Variable, Value>() { { y, null } };
            y.Evaluate(inputDataMap3, outputDataMap, DeviceDescriptor.CPUDevice);
            var evaluation = outputDataMap[y].GetDenseData<float>(y)[0][0];


            // True negative
            if (expectedOutput == 0 && evaluation < 0.5)
            {
                Console.WriteLine($"The passanger did not survive.\r\n" +
                    $"Expected: {expectedOutput}, calculated: {evaluation}");
            }
            // True positive
            else if (expectedOutput == 1 && evaluation >= 0.5)
            {
                Console.WriteLine($"The passanger survived.\r\n" +
                    $"Expected: {expectedOutput}, calculated: {evaluation}");
            }
            // False negative or false positive
            else
            {
                Console.WriteLine($"Calculation failed!\r\n" +
                    $"Expected: {expectedOutput}, but got: {evaluation}");
            }
        }

        private static double Evaluate(int inputSize, int outputSize, int batchSize, List<float> training_inputs, List<float> training_outputs, int trainingCount, Variable x, Function y)
        {
            Variable yt = Variable.InputVariable(new int[] { outputSize }, DataType.Float);

            Function y_rounded = CNTKLib.Round(y);
            Function y_yt_equal = CNTKLib.Equal(y_rounded, yt);
            Evaluator evaluator = CNTKLib.CreateEvaluator(y_yt_equal);

            double sumEval = 0;
            for (int batchI = 0; batchI < trainingCount / batchSize; batchI++)
            {
                Value x_value = Value.CreateBatch(x.Shape, training_inputs.GetRange(batchI * batchSize * inputSize, batchSize * inputSize), DeviceDescriptor.CPUDevice);
                Value yt_value = Value.CreateBatch(yt.Shape, training_outputs.GetRange(batchI * batchSize * outputSize, batchSize * outputSize), DeviceDescriptor.CPUDevice);
                var inputDataMap = new UnorderedMapVariableValuePtr()
                    {
                        { x, x_value },
                        { yt, yt_value }
                    };

                sumEval += evaluator.TestMinibatch(inputDataMap, DeviceDescriptor.CPUDevice) * batchSize;
            }
            return sumEval / trainingCount;
        }
    }
}
