using System;
using MyMediaLite.RatingPrediction;
using AutoDiff;

namespace MFAutoDiff
{
	public class BaselineAutoDiff : UserItemBaseline
	{
		static double[] GradientDescent(ICompiledTerm func, double[] init, double stepSize, int iterations)
		{
		    // clone the initial argument
		    var x = (double[])init.Clone();
		
		    // perform the iterations
		    for (int i = 0; i < iterations; ++i)
		    {
		        // compute the gradient
		        var gradient = func.Differentiate(x).Item1;
		
		        // perform a descent step
		        for (int j = 0; j < x.Length; ++j)
		            x[j] -= stepSize * gradient[j];
		    }
		
		    return x;
		}		
			
		public override void Train()
		{
			InitModel();
			
            // define variables
            var avg = Ratings.Average;
            var ub  = new Variable();
            var ib  = new Variable();
			var r   = new Variable();

            // define function to minimize
			var func = TermBuilder.Power(avg + ub + ib - r, 2);
			var func_compiled = func_min.Compile();
			
			double learn_rate = 0.0001;
			
			
			for (int i = 0; i < NumIter; i++)
				foreach (int j in Ratings.RandomIndex)
				{
					// get data
					var data = new double[] { user_bias[Ratings.Users[j]], item_bias[Ratings.Items[j]], Ratings[j] };
				
					// compute gradient
					var gradient = func_compiled.Differentiate(data);
				
					// update
					user_bias[Ratings.Users[j]] -= learn_rate * gradient[0];
					item_bias[Ratings.Items[j]] -= learn_rate * gradient[1];
				}
		}
		
		public override double Predict(int user_id, int item_id)
		{
			double user_bias = (user_id < user_biases.Length && user_id >= 0) ? user_biases[user_id] : 0;
            double item_bias = (item_id < user_biases.Length && item_id >= 0) ? item_biases[item_id] : 0;
            
			double result = global_average + user_bias + item_bias;
						
            if (result > MaxRating)
            	result = MaxRating;
            if (result < MinRating)
                result = MinRating;
            return result;
		}
	}
}

