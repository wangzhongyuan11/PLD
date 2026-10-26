// app.js — PLD 三模型分割演示小程序
App({
  globalData: {
    env: "pld-d7gzvyxp10bdc9437",
  },

  onLaunch: function () {
    if (!wx.cloud) {
      console.error("请使用 2.2.3 或以上的基础库以使用云能力");
    } else {
      wx.cloud.init({
        env: this.globalData.env,
        traceUser: true,
      });
    }
    this._privacyListeners = [];
    this._privacyResolve = null;
    const self = this;
    if (wx.onNeedPrivacyAuthorization) {
      wx.onNeedPrivacyAuthorization(function (resolve) {
        self._privacyResolve = resolve;
        self._privacyListeners.forEach(function (fn) { fn(true); });
      });
    }
  },

  registerPrivacy: function (fn) {
    if (this._privacyListeners.indexOf(fn) < 0) this._privacyListeners.push(fn);
    var self = this;
    return function () {
      var i = self._privacyListeners.indexOf(fn);
      if (i >= 0) self._privacyListeners.splice(i, 1);
    };
  },

  agreePrivacy: function () {
    this._privacyListeners.forEach(function (fn) { fn(false); });
    if (this._privacyResolve) {
      this._privacyResolve({ event: "agree", buttonId: "agree-btn" });
      this._privacyResolve = null;
    }
  },

  refusePrivacy: function () {
    this._privacyListeners.forEach(function (fn) { fn(false); });
    if (this._privacyResolve) {
      this._privacyResolve({ event: "disagree" });
      this._privacyResolve = null;
    }
  },
});
