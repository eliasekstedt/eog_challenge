
# pinned
only by understanding what my model is doing can i understand how to make it better. avoid the temptation of making blind changes to hopefully increment your way up the leaderboard.


# improve faster
before testing different hyperparameters, make predictions about what the effect will be on training and test performance. for example, since the model is trainperformance seems to converge well but not test performance, adding more layers to the fully connected segment would not likely improve test performance. if anything there may be too many parameters in the model as it seems to capture noise well.

read about what competition winners of similar competitions say was key to their success. if the methods are applicable to your problem, understand how the methods work by reading the methods article and implement the method in your code. suggestions for core questions that i should begin to answer for each method:

    is the method applicable to my current problem
    where does the method fit in to my system (preprocessing, model architecture, etc)

# current participants
    day active  enrolled   best_score   my_score    my_position first
    07/27   21  153 6.394   8.271   10  cloudyy 
    07/28   23  165 6.390   8.271   10  cloudyy
    07/29   25  175 6.390   7.537   10  cloudyy
    08/01   30  201 6.390   7.537   10  cloudyy
    08/02   30  207 6.390   7.537   12  cloudyy
    08/03   32  219 6.390   7.165   12  cloudyy
    08/04   33  230 6.390   6.889   9   cloudyy
    08/05   35  241 6.390   6.889   9   cloudyy
    08/06   38  248 6.343   6.889   10  quanvm4
    08/07   40  255 6.312   6.889   10  quanvm4
    08/08   41  259 6.312   6.889   10  quanvm4
    08/09   43  269 6.312   6.889   11  quanvm4
    08/10   44  277 6.312   6.889   11  quanvm4
    08/11   48  286 6.281   6.889   13  quanvm4
    08/12   49  291 6.281   6.889   13  quanvm4
    08/13   50  293 6.260   6.889   13  quanvm4
    08/14   53  300 6.233   6.889   14  quanvm4
    08/15   54  321 6.233   6.889   16  quanvm4
    08/16   56  340 6.233   6.889   17  quanvm4
    08/17   57  350 6.233   6.889   17  quanvm4
    08/18   57  357 6.233   6.889   17  quanvm4
    08/19   58  366 6.233   6.889   17  quanvm4
    08/20   60  371 6.233   6.889   17  quanvm4
    08/21   63  374 6.223   6.889   17  quanvm4
    08/22   65  378 6.223   6.889   17  quanvm4
    08/23   67  385 6.223   6.889   17  quanvm4
    08/24   67  394 6.223   6.889   18  quanvm4
    08/25   70  401 6.223   6.889   18  quanvm4
    08/26   70  413 6.223   6.889   18  quanvm4
    08/27   70  420 6.223   6.889   19  quanvm4



APM KEYWORD: bayesian inference algorithm
# order of implementation for future projects (not final, keep adjusting)
    architecture (basic components e.g, conv, fc integration)
    batch normalization
    batch size
    weight decay
    dropout


# to learn
    torch.compile
    reproducibility in pytorch (before grid search)
    batch normalizatin seems important. i need to understand it in greater detail.

# factors that may impact the result and may therefor need to be evaluated and amended

    image size is currently forced as 128x128 no matter the different original image sizes

    no matter the contextual categories belonging to an image, the same image model is used for all. the model does not recieve feedback on damage type

# possible implementations
    sky cropping (targeted or not)

    information about original image size (difference between original and resized)

    dimensions of fully connected segment
        test fully connected segment independently (JustFC). try to extract the weights from the relevant layer after a model with both conv and fc have been trained. take the weights from the output of conv to simulate this output


# end tweeks defined (possibly ordered)

    grid search or any of its variations (Bayesian Optimization)

    running epochs more than 15
    
    skipping accuracy* measure to let the code run in the allowed time intervall and give all labeled data for parameter tuning


