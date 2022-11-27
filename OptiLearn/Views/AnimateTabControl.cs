using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Metadata;
using Avalonia.Styling;
using System;

namespace OptiLearn
{
    [PseudoClasses(":normal")]
    public class AnimateTabControl : TabControl, IStyleable
    {
        Type IStyleable.StyleKey => typeof(TabControl);

        public AnimateTabControl()
        {
            PseudoClasses.Add(":normal");
            this.GetObservable(SelectedContentProperty).Subscribe(OnContentChanged);
        }

        private void OnContentChanged(object? obj)
        {
            if (AnimateOnChange && obj != null)
            {
                PseudoClasses.Remove(":normal");
                PseudoClasses.Add(":normal");
            }
        }

        public bool AnimateOnChange
        {
            get => GetValue(AnimateOnChangeProperty);
            set => SetValue(AnimateOnChangeProperty, value);
        }

        public static readonly StyledProperty<bool> AnimateOnChangeProperty =
            AvaloniaProperty.Register<AnimateTabControl, bool>(nameof(AnimateOnChange), true);
    }
}