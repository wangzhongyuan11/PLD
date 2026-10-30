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
    this._privacyResolvers = [];
    const self = this;
    if (wx.onNeedPrivacyAuthorization) {
      wx.onNeedPrivacyAuthorization(function (resolve) {
        self._privacyResolvers.push(resolve);
        self._privacyListeners.forEach(function (fn) { fn(true); });
      });
    }
  },

  registerPrivacy: function (fn) {
    if (this._privacyListeners.indexOf(fn) < 0) this._privacyListeners.push(fn);
    if (this._privacyResolvers.length) fn(true);
    var self = this;
    return function () {
      var i = self._privacyListeners.indexOf(fn);
      if (i >= 0) self._privacyListeners.splice(i, 1);
    };
  },

  agreePrivacy: function () {
    this._privacyListeners.forEach(function (fn) { fn(false); });
    var pending = this._privacyResolvers.splice(0);
    pending.forEach(function (resolve) { resolve({ event: "agree", buttonId: "agree-btn" }); });
  },

  refusePrivacy: function () {
    this._privacyListeners.forEach(function (fn) { fn(false); });
    var pending = this._privacyResolvers.splice(0);
    pending.forEach(function (resolve) { resolve({ event: "disagree" }); });
  },
});
