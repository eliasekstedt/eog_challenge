# 26/07/23
should have thought about all the things that would be different for this project comparet to the things ive done previously and therefor what needs to change to accomodate. that way i would not have been surprised when i get errors. for example, im now dealing with a regression problem which is a first for me in pytorch.

things are up and running, results are not yet great.
i should find a way to let the model account for the contextual data given in the csv.
consider doing the same with original image size.
should also find a proper architecture.
could i find the network used for previous years and build upon it? could i find the article for tips not just about the architecture. talking if they split the data into different categories or if they use my idea for accounting for category data, and so on.

next step is to do some more research on how this problem, or problems like this have been handeled in the past.

# 30/0723
current challenge:
find a way to implement targeted cropping. need to find a model pretrained on a suitable dataset. since all pytorches models for semantic segmentation seem to be trained on COCO it appears i can only get them from github in a way that requires some knowledge i do not yet have.

im in a decent position on the leaderboard. i am confident in how i have structured my model, the general structure and hyper parameters i have choosen. the whole assembly can be broken down into dataprocessing and model developing. further plans for each are:

    dataprocessing
        implement cropping, targeted or otherwise
            through pretrained segmentation model
            indiscriminant cropping
            unsupervised approach(?)

        find a way to handle varying input sizes in images
            one for all

    model developing
        fine-tune fc model structure separately from convolutional model

    hyperparameter tuning
        save fine tuning to when all else is ready
        begin setting the hyperparameters that act early in the process, but tune the set of parameters iteratively.


considering categorizing hyperparameters. after all, they impact the prediction quality in many different ways

currently doing the required work to get fc running independently from conv so that i can develop fc faster.

# 31/07/23
i dont seem to be able to get the training of the fully connected layers apart from the rest of the model working the way i want. maybe better to train the full model and save the model state, then load that model state and train with different hyperparameters and observe the results. however, this will not allow me to try the full range of configurations that i had intended. my best guess as to what went wrong is that the fc layers were synchronized with the rest of the model. for the output rest of the model to produce good results the parameters of the fc layers must be close in sync. this synchronization is very unlikely if both parts of the model did not arrive at them from the begining of training. perhaps i could still make it work if i also saved the state of the fc layers so that these layers are in sync at the beginning of fc testing.

to improve faster: read about what competition winners of similar competitions say was key to their success. if the methods are applicable to your problem, understand how the methods work by reading the methods article and implement the method in your code. what are core questions that i should begin to answer for each method?

# 01/08/23
i think that the reason that the model is performing worse now than it did a few days ago is that i created new train and test splits. today i found that performance gets almost on par with what it used to be when in reduce the batch size to 100 where it used to be 200. i also tried 600. this made test performance significantly worse eventhough it is suposed to get better. (train performance appears about the same). it was suggested that a larger batch size can make it harder for the model to get out of local minima and i think this is what has happened here. gonna run different values of dropout_rate. i predict that for the best rate train performance will not suffer significantly while test performance will improve.

# 02/08/23
test performance improves to a point around eight to seven where it then plateaus. train performance continues unhindered. this is the behaviour ive been trying to change, mostly from increasing weight decay. never got it below 7. infact, although i gave the implementation of weight decay credit for early success, it seems that withing the range of values ive tried (1e-7 throught 2e-5), this parameter does not affect either curve significantly.

# 03/08/23
using batch normalization before every layer in the fully connected model seems to have solved the overfitting problem that always turned up around epoch 9. i think that a big reason why my model is able to perform at this level is that i take small steps and evaluate after each thing i implement, no matter how small. continuing with this i think is especially important right now, because i have many different things to test in mind: changing the resolution of the image, setting batch normalization between linear and relu rather than between layers, using the more complex model with dropout, etc. some time around here i should probably also try at least a two-fold cross validation. question is, when is the right time for the time trade-off.

# 04/08/23
switched to more complex model but results have not been impacted greatly. tried different combinations of image size and batch size but even 4x the number of pixels does not seem to matter. going to have a look at how the distribution in prediction values are compared to the distribution of labels. if it is as i suspect that predictions tend to group closer to zero, then maybe it could help to modify the loss function with an extra penalty for the greater the prediction value to even predictions out.

# 05/08/23
been looking at how extent is distributed compared to predictions. for some reason where the label is 10, the model guesses 20 in almost all cases and rarely correctly guesses 10. been working on implementing a weighed loss function but will finish tomorrow. 

# 07/08/23
maybe i can find a better place for dropout. perhaps right after the layer exiting res18, right before integration with context features... actually, now im thinking dropout doesnt work with regression problems, though gtp disagrees.

taking a break now. when i come back, the first thing i need to do is to make my own version of the confusion matrix, or something that shows that same information. consider the distributions plot as a template as it is simpler and has the same information. still need to improve it though to make it more intuitive. consider normalizing but saving the information as a weight for each extent class where the weight is the number of datapoints in each distribution. once i can see the type of misclassifications that are made i can begin to look for why. for example by looking at images of misclassified datapoints. This takes priority over implementing cross validation, weighted cost function and getting a functional dropout layer.

should also have a look at the links provided by the dlia cheat sheet.

# 20/08/23
no circos solution is a good substitute for a good confusion matrix

need to define modules in higher resolution. what are the subprocesses of the entire system from prepping datafolds to producing the final submission file. what are the inputs and outputs to each module and are there any factors that make a difference here. what evaluation metrics to i want to produce and based on what data. right now everything after training and performance plotting is a mess because i dont know what files are used in which process and what the requirements on those files are. there is a huge difference between creating the submission file and producing evaluation metrics, yet in my mind there is still significant overlap between them. for one thing they are based on completely separate data. here are some possible areas to focus on:
* dataprep (with consideration for potential plots)
* dataloadeing (cross val, btw crossval.py could be written way better)
* model training
* use models of both folds in the fork
    - produce evaluation metrics based on data of both folds
    - produce submission file
    (here define exactly where the fork should come)


overall the code complexity of this project has grown a bit out of control and i should try to reduce it where possible. consider one tool file for each module. consider one util folder for each module. be careful not to overdo it.

create a map of defined modules.

maybe spend five minutes looking up how to suppress warnings about deprecation of 'pretrained' parameter

once this is complete i can focus on implementing bayesian parameter optimization. i do worry however that stochasisity will influence its ability to find optimal parameters. (if i knew as much as i should about bayesian modeling i might have had a way to solve the problem).

# 21/08/23
thinking about overhauling code with more separation of concerns. got to define the concerns to be separated first though. before i do that i should implement bayes optimization so that i have something that can run for a long time while i do that.

implemented early stopping that stops training (not yet tested). will save time during optimization.

# 22/08/23
some thoughts:
- what is my resnet model pretrained on and is it beneficial at this point
- label balance in labeled vs unlabeled (val) data. 0 guessing is ~19-20 for val. same for labeled? consider balansing training data rather than using a penalty.
- enforcing deterministic training is probably essential for quality parameter optimization.

things i want to do right now:
- see example code from (kaggle) competitions
- begin learning about XAI
- enable deterministic training
- rewrite code with more separation of concerns
- get a sense of label balance in labeled v unlabeled datasets
    * conclusion: both sets appear statistically equivalent

# 23/08/23
the conclusion that im drawing from the gaussian process optimizing weight decay is that this parameter no longer makes any difference. i suspect that it has to do with either an overuse of batch normalization or vanishing gradients. it could also be that i dont let it run for enough epochs where it is allowed to prevent overfitting which is what it is supposed to do.

# 24/08/23
lesson: because of the contextual features provided in addition to the images it might have been good to start with simpler methods, e.g rule based systems or just fc layers excluding the images. in the case of rule based systems this could also helped me better understand the data. it would have been interesting to see the improvement from just guessing the optimal value based on labeled data, to the best predictions of the simpler method, to the improvement when i finally implement the neural network.


# _/08/23
# _/08/23

