package  
{
	import flash.display.Sprite;
	import flash.display.Stage;
	import flash.filters.GlowFilter;
	import flash.filters.BlurFilter;
	import flash.utils.Timer;
	import flash.events.TimerEvent;
	
	public class LoadScreenContainer extends Sprite
	{
		private var mStage:Stage;
		private var bg:Sprite;
		private var loadScreen:LoadScreen;
		private var alphaTimer:Timer;
		private var isVisible:Boolean = true;
		
		public function LoadScreenContainer(stage:Stage) 
		{
			super();
			
			this.mStage = stage;
			Initialize();
		}
		
		private function Initialize() : void
		{
			bg = new Sprite();
			bg.graphics.clear();
			bg.graphics.beginFill(0, 1);
			bg.graphics.drawRect(0, 0, mStage.stageWidth, mStage.stageHeight);
			bg.graphics.endFill();
			
			loadScreen = new LoadScreen();
			loadScreen.x = mStage.stageWidth / 2 - loadScreen.width * 1.25;
			loadScreen.y = mStage.stageHeight / 2 - loadScreen.height * 0.75;
			var filter:GlowFilter = new GlowFilter(0xFFFF0000, 1, 4, 4, 2, 1);
			loadScreen.filters = [filter];
			
			this.alphaTimer = new Timer(33, 16);
			
			this.addChild(bg);
			this.addChild(loadScreen);
		}
		
		public function get Visible() : Boolean
		{
			return this.isVisible;
		}
		
		public function set Visible(value:Boolean) : void
		{
			if(this.isVisible != value)
			{
				alphaTimer.reset();
				if(alphaTimer.hasEventListener(TimerEvent.TIMER)) alphaTimer.removeEventListener(TimerEvent.TIMER, TimerHandler);
				alphaTimer.addEventListener(TimerEvent.TIMER, TimerHandler, false, 0, true);
				alphaTimer.start();
				
				this.isVisible = value;
			}
		}
		
		private function TimerHandler(e:TimerEvent) : void
		{
			if(this.isVisible)
			{
				this.alpha = Math.min(this.alpha + 0.0625, 1);
			}
			else
			{
				this.alpha = Math.max(this.alpha - 0.0625, 0);
			}
		}

	}
	
}
