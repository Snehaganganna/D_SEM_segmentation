
import matplotlib.pyplot as plt
import os


def plot_training_history(history, save_dir='.'):
    """Plot training and validation metrics"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot F1 score
    ax2.plot(history['train_f1'], label='Training F1')
    ax2.plot(history['val_f1'], label='Validation F1')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Training and Validation F1 Score')
    ax2.legend()
    ax2.grid(True)
    
    # Plot learning rate if available
    if 'learning_rate' in history and history['learning_rate']:
        ax3.plot(history['learning_rate'], 'g-')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_yscale('log')
        ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()


def plot_augmentation_performance(history, save_dir='.'):
    """Plot performance metrics for each augmentation type"""
    plt.figure(figsize=(12, 6))
    
    # Plot F1 score for each augmentation type
    for aug_name in history['aug_performance']:
        if history['aug_performance'][aug_name]:  # Check if there's data for this augmentation
            plt.plot(history['aug_performance'][aug_name], label=aug_name)
    
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Performance by Augmentation Type')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'augmentation_performance.png'))
    plt.close()

    # Also create a bar plot for final epoch performance
    plt.figure(figsize=(10, 6))
    
    final_scores = {}
    for aug_name in history['aug_performance']:
        if history['aug_performance'][aug_name]:
            final_scores[aug_name] = history['aug_performance'][aug_name][-1]
    
    if final_scores:
        plt.bar(final_scores.keys(), final_scores.values())
        plt.xlabel('Augmentation Type')
        plt.ylabel('Final F1 Score')
        plt.title('Final Epoch Performance by Augmentation Type')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        
        plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'final_augmentation_performance.png'))
    plt.close()
 
   
def plot_augmentation_performance(history, save_dir='.'):
    plt.figure(figsize=(12, 6))
    
    for aug_name in history['aug_performance']:
        plt.plot(history['aug_performance'][aug_name], label=aug_name)
    
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Performance by Augmentation Type')
    plt.legend()
    plt.grid(True)
    plt.savefig('augmentation_performance.png')
    plt.close()