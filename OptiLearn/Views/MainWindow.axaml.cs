using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Layout;
using Avalonia.Media;
using Avalonia.Threading;
using Material.Styles.Themes;
using Microsoft.ML.Models.BERT;
using OptiLearn.ViewModels;
using ReactiveUI;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Zexsoft;
using System.Speech.Synthesis;
using System;
using System.Globalization;
using Avalonia.Interactivity;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Timers;

namespace OptiLearn.Views
{
    public partial class MainWindow : Window
    {
        Course currentCourse;
        User currentUser = new User();

        public int pomodoroCount = 0;
        public int pomodoroRemaining = 0;
        public Timer pomodoroTimer;

        BertModel modelQuestion;
        ObservableCollection<Conversation> assistantChat { get; set; } = new();

        public MainWindow()
        {
            InitializeComponent();

            currentCourse = new Course("World War I", """World War I or the First World War, often abbreviated as WWI or WW1, and referred to by some Anglophone authors as the "Great War" or the "War to End All Wars", was a global conflict which lasted from 1914 to 1918, and is considered one of the deadliest conflicts in history. Belligerents included much of Europe, the Russian Empire, the United States, and the Ottoman Empire, with fighting occurring throughout Europe, the Middle East, Africa, the Pacific, and parts of Asia. An estimated 9 million soldiers were killed in combat, plus another 23 million wounded, while 5 million civilians died as a result of military action, hunger, and disease. Millions more died in genocides within the Ottoman Empire and the 1918 influenza pandemic, which was exacerbated by the movement of combatants during the war. Prior to 1914, the European great powers were divided between the Triple Entente, comprising France, Russia, and Britain, and the Triple Alliance, containing Germany, Austria-Hungary, and Italy. Tensions in the Balkans came to a head on 28 June 1914 following the assassination of Archduke Franz Ferdinand, the Austro-Hungarian heir, by Gavrilo Princip, a Bosnian Serb. Austria-Hungary blamed Serbia, which led to the July Crisis, an unsuccessful attempt to avoid conflict through diplomacy. On 28 July 1914, Austria-Hungary declared war on Serbia, and Russia came to the latter's defence. By 4 August, the system of entangling alliances drew in Germany, France, and Britain, along with their respective colonies, although Italy initially remained neutral. In November 1914, the Ottoman Empire, Germany, and Austria-Hungary formed the Central Powers, and on 26 April 1915, Italy joined Britain, France, Russia, and Serbia as the Allies of World War I. Facing a war on two fronts, German strategy in 1914 was to concentrate its forces on defeating France in six weeks, before moving them to the Eastern Front and doing the same to Russia. However, this was defeated at the Marne in September 1914, and the year ended with the two sides facing each other along the Western Front, a continuous series of trenches stretching from the English Channel to Switzerland. The frontlines in the West changed little until 1917, while the Eastern Front was far more fluid, with both Austria-Hungary and Russia gaining and losing large swathes of territory. Other significant theatres included the Middle East, Italy, Asia Pacific, and the Balkans, which drew Bulgaria, Romania, and Greece into the war. Over the course of 1915, both Russia and Austria-Hungary suffered enormous casualties in the East, while Allied offensives in Gallipoli and on the Western Front ended in failure. German attacks at Verdun in 1916, and a combined Franco-British offensive on the Somme cost participants heavy losses for limited strategic gains, while the Russian Brusilov offensive ground to a halt after early success. By 1917, Russia was on the verge of revolution, and the failure of the French Nivelle Offensive, as well as costly British attacks in Flanders, meant all belligerents were short of manpower and under severe economic stress. Shortages caused by the Allied naval blockade led Germany to initiate unrestricted submarine warfare in early 1917, bringing the previously-neutral United States into the war on 6 April 1917. In Russia, the Bolsheviks seized power in the 1917 October Revolution and exited the war with the March 1918 Treaty of Brest-Litovsk, freeing up large numbers of German troops. The German General Staff used these additional resources to launch the March 1918 offensive, which was halted by stubborn Allied defence, heavy casualties, and supply shortages. When the Allies began the Hundred Days Offensive in August, the Imperial German Army continued to fight hard but could only slow the advance, not stop it. Towards the end of 1918, the Central Powers began to collapse; Bulgaria signed an armistice on 29 September, followed by the Ottomans on 31 October, then Austria-Hungary on 3 November. Isolated, facing the German Revolution at home and a military on the verge of mutiny, Kaiser Wilhelm abdicated on 9 November, and the new German government signed the Armistice of 11 November 1918, bringing the conflict to a close. The Paris Peace Conference of 1919–1920 imposed various settlements on the defeated powers, with the best-known of these being the Treaty of Versailles. The dissolution of the Russian Empire in 1917, the German Empire in 1918, the Austria-Hungarian Empire in 1920, and the Ottoman Empire in 1922, led to numerous uprisings and the creation of independent states, including Poland, Czechoslovakia, and Yugoslavia. For reasons that are still debated, failure to manage the instability that resulted from this upheaval during the interwar period ended with the outbreak of World War II in September 1939.""");

            // DataContext
            grCourses.DataContext = currentCourse;
            tMain.DataContext = currentUser;
            lbChat.Items = assistantChat;
            
            // Models
            if (File.Exists("Model/bert-question.onnx"))        // BERT Question
            {
                BertModelConfiguration modelCfgQuestion = new BertModelConfiguration()
                {
                    VocabularyFile = "Model/vocab.txt",
                    ModelPath = "Model/bert-question.onnx"
                };

                modelQuestion = new BertModel(modelCfgQuestion);
                modelQuestion.Initialize();
            }
            else
            {
                tbAssistant.Text = "AI Assistant unavailable.";
                tbAssistant.IsEnabled = false;
            }
        }

        private void tbAssistant_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Enter)
            {
                //pbResponse.IsVisible = true;
                //txResponse.IsVisible = false;
                assistantChat.Add(new Conversation(tbAssistant.Text, true));

                Dispatcher.UIThread.Post(() => { }, DispatcherPriority.MaxValue);

                //txResponse.Text = ;
                assistantChat.Add(new Conversation(QuestionAI(tbAssistant.Text), false));

                lbChat.ScrollIntoView(assistantChat[assistantChat.Count - 1]);
                tbAssistant.Text = string.Empty;

                e.Handled = true;
            }
        }

        SpeechSynthesizer synthesizer = new SpeechSynthesizer();

        private void btNarrator_Click(object sender, RoutedEventArgs e)
        {
            if (synthesizer.GetCurrentlySpokenPrompt() != null)
            {
                synthesizer.SpeakAsyncCancelAll();
                return;
            }

            try
            {
                synthesizer.SetOutputToDefaultAudioDevice();

                PromptBuilder builder = new PromptBuilder();
                builder.StartVoice(currentUser.Region);
                builder.AppendText(currentCourse.Content);
                builder.EndVoice();

                synthesizer.SpeakAsync(builder);
            }
            catch
            {
                tbAssistant.Text = "Narrator is not supported on your platform.";
            }
        }

        private void ckSession_PropertyChanged(object sender, AvaloniaPropertyChangedEventArgs e)
        {

        }

        // AI MODELS

        public string QuestionAI(string query)
        {
            string[] sentences = Regex.Split(currentCourse.Content, @"(?<=[\.!\?])\s+");

            QSearch search = new QSearch(0, 0) { UseTokenLink = true, MatchCase = false };
            List<string> res = search.ProcessQuery(sentences.ToList(), query);
            string context = (res.Count() > 0 ? res.Aggregate((sum, val) => sum + ". " + val) : "") + sentences.Aggregate((sum, val) => sum + ". " + val);
            if (context.Length > 256) context = context.Remove(256);

            var (tokens, probability) = modelQuestion.Predict(context, query);
            return tokens.Aggregate((sum, val) => sum + " " + val);
        }
    }

    public class Conversation : ViewModelBase
    {
        public string Text;

        public bool Side;

        public Conversation(string text, bool side) 
        {
            Text = text;

            Side = side;
        }

        public string txText
        {
            get => Text;
            set => this.RaiseAndSetIfChanged(ref Text, value);
        }

        public SolidColorBrush? SideColor
            => Side ? (SolidColorBrush)Application.Current.FindResource("MaterialDesignBody") : (SolidColorBrush)Application.Current.FindResource("TertiaryHueMidBrush");

        public HorizontalAlignment Orientation
        {
            get => Side ? HorizontalAlignment.Right : HorizontalAlignment.Left;
        }
    }
}
